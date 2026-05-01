import argparse
from fileinput import filename
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_clip_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolynomialLRWarmup
from partial_fc_v2 import PartialFC_V2
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.clip_verifications_utils import  ClipVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
import torch.nn.functional as F

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

def clip_face_loss(emb_partial_face, emb_full_face, temperature=0.07):
    # emb_partial, emb_full: [N, D]

    emb_partial_face = F.normalize(emb_partial_face, dim=1)
    emb_full_face    = F.normalize(emb_full_face, dim=1)

    logits = emb_partial_face @ emb_full_face.T
    logits = logits / temperature

    targets = torch.arange(len(logits), device=logits.device)

    loss_p2f = F.cross_entropy(logits, targets)
    loss_f2p = F.cross_entropy(logits.T, targets) #TODO: why is it not the same value as loss_p2f?

    return 0.5 * (loss_p2f + loss_f2p)


def unnormalize(img_tensor):
    mean = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device).view(-1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=img_tensor.device).view(-1, 1, 1)
    return img_tensor * std + mean


def save_image_grid(batch_tensor, filename="grid.jpg", nrow=8):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # batch_tensor: [B, C, H, W]
    grid = make_grid(batch_tensor, nrow=nrow, padding=2)
    grid = unnormalize(grid).clamp(0, 1)
    
    to_pil = ToPILImage()
    grid_img = to_pil(grid)
    grid_img.save(filename)
    print(f"Saved image grid to {filename}")

rank = 0
local_rank = 0
world_size = 1


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    train_loader = get_clip_dataloader(
        cfg.root_pf, cfg.root_ff,
        local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.dali_aug,
        cfg.seed,
        cfg.num_workers
    )

    # ------------ load teacher model -------------
    teacher_backbone = get_model(
        cfg.teacher_network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    teacher_backbone.eval()
    teacher_state_dict = torch.load(cfg.teacher_model_path)
    teacher_backbone.load_state_dict(teacher_state_dict)
    teacher_backbone.eval()
    # ---------------------------------------------

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    # backbone._set_static_graph()

    if cfg.optimizer == "sgd":
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)

    start_epoch = 0
    global_step = 0

    if cfg.resume:
        try:
            dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
            start_epoch = dict_checkpoint["epoch"]
            global_step = dict_checkpoint["global_step"]
            backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
            opt.load_state_dict(dict_checkpoint["state_optimizer"])
            lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
            del dict_checkpoint
        except:
            print("No checkpoint found, training from scratch.")
            logging.info("No checkpoint found, training from scratch.")

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    clip_verification = ClipVerification(
        val_targets=cfg.val_targets, train_targets=cfg.train_targets, 
        summary_writer=summary_writer, wandb_logger = None, work_dir=cfg.output
    )

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    loss_am = AverageMeter()

    best_model_acc = 0.0

    ### training loop ###
    for epoch in range(start_epoch, cfg.num_epoch):
        
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for _, (full_face_img, partial_face_img, labels )  in enumerate(train_loader):
            # if _ >= 100:
            #     break
            if _ == 0 :
                save_image_grid(partial_face_img, f"{cfg.output}/partial_face_img_batch_grid.jpg", nrow=4)
                save_image_grid(full_face_img, f"{cfg.output}/full_face_img_batch_grid.jpg", nrow=4)

            global_step += 1
            emb_partial = backbone(partial_face_img)
            emb_full = teacher_backbone(full_face_img).detach()

            loss = clip_face_loss(emb_partial, emb_full, temperature=cfg.temperature)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                opt.zero_grad()
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():                   
                loss_am.update(loss, 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
        logging.info("Starting verification process")
        ver_accs, rank1 = clip_verification(
            backbone_partial=backbone,
            backbone_full=teacher_backbone,
            global_step=global_step,
            epoch=epoch,
        )
        logging.info("Epoch {}: Rank-1 Acc {:.5f}".format(epoch, rank1))
        logging.info("Epoch {}: Verification Accuracies: {}".format(epoch, ver_accs))


        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        
        ## assuming acc_list[0] is the validation accuracy
        try:
            if ver_accs >= best_model_acc:
                best_model_acc = ver_accs
                # best_model_triplet_losses = ver_triplet_losses[0]
                torch.save(backbone.state_dict(), os.path.join(cfg.output, "best_model.pt"))
        except:
            print("Warning: Could not save best model.")

        #save the model checkpoint every epoch
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.state_dict(), path_module)
    #save the model checkpoint at the end of training
    path_module = os.path.join(cfg.output, "model.pt")
    torch.save(backbone.state_dict(), path_module)
    results_dict = {
        "best_acc": best_model_acc,
    }
    return results_dict


def train(args):
    results_dict = main(args)
    return results_dict


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config",default="configs/exp_glint360k_vit_l_clip.py", type=str, help="py config file")
    results_dict = main(parser.parse_args())
    print(f"Best Acc: {results_dict['best_acc']:.5f}")