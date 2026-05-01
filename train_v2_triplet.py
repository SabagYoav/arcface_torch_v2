import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from backbones import get_model
from dataset import get_dataloader, get_batch_triplet_dataloader
from losses import BatchAllTripletLoss
from lr_scheduler import PolynomialLRWarmup
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
from utils.utils_logging import AverageMeter, init_logging
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

from utils.data_utils import save_batch_grid

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )
    
    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        # Sign in to wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print("WandB Key must be provided in config file (base.py).")
            print(f"Config Error: {e}")
        # Initialize wandb
        run_name = datetime.now().strftime("%y%m%d_%H%M") + f"_GPU{rank}"
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name, 
                notes = cfg.notes) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print("WandB Data (Entity and Project name) must be provided in config file (base.py).")
            print(f"Config Error: {e}")
    # train_loader = get_dataloader(
    #     cfg.rec,
    #     local_rank,
    #     cfg.batch_size,
    #     cfg.dali,
    #     cfg.dali_aug,
    #     cfg.seed,
    #     cfg.num_workers
    # )
    train_loader = get_batch_triplet_dataloader(
            cfg.rec,
            local_rank,
            cfg.batch_size,
            cfg.dali,
            cfg.dali_aug,
            cfg.seed,
            cfg.num_workers,
            num_instances=cfg.num_instances,
        )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    triplet_margin = getattr(cfg, 'triplet_margin', 0.3)
    triplet_loss_fn = BatchAllTripletLoss(margin=triplet_margin)

    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(
            params=backbone.parameters(),
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params=backbone.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step,
        #power=1.0,
        #min_lr=cfg.lr / 100.0,
        )

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, 
        summary_writer=summary_writer, wandb_logger = wandb_logger
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    best_model_acc = 0.0
    #first verification before training

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)

        for _, (img, local_labels) in enumerate(train_loader):
            if _ == 0 and epoch == start_epoch:
                save_batch_grid(img, save_path=os.path.join(cfg.output, f"inspection_batch_grid.jpg"), nrow=cfg.num_instances, normalize_from_minus1_1=True)

            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = triplet_loss_fn(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():     
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
        with torch.no_grad():  
            ver_accs = callback_verification(global_step, backbone, epoch)
            # Write epoch-level training loss
            if summary_writer is not None:
                summary_writer.add_scalar('Train_Loss/epoch', loss_am.avg, epoch)
            loss_am.reset()

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
       
        ## assuming acc_list[0] is the validation accuracy
        try:
            if ver_accs[0] >= best_model_acc:
                best_model_acc = ver_accs[0]
                # best_model_triplet_losses = ver_triplet_losses[0]
                torch.save(backbone.module.state_dict(), os.path.join(cfg.output, "best_model.pt"))
        except:
            print("Warning: Could not save best model.")

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

    results_dict = {
        "best_acc": best_model_acc,
        # "arcface_loss": loss_am.avg,
        # "triplet_loss": best_model_triplet_losses
    }
    return results_dict

def train(args):
    results_dict = main(args)
    return results_dict


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config",default="configs/exp_glint360k_roi_100_r50_triplet.py", type=str, help="py config file")
    results_dict = main(parser.parse_args())
    print(f"Best Acc: {results_dict['best_acc']:.5f}")