import logging
import os
import time
from typing import List

import torch

from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed


class CallBackVerification(object):
    
    def __init__(
        self,
        val_targets,
        rec_prefix,
        summary_writer=None,
        image_size=(112, 112),
        wandb_logger=None,
        subset_train_num_ids=None,
        subset_seed=42,
    ):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.subset_train_num_ids = subset_train_num_ids
        self.subset_seed = subset_seed
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger

        self.ver_accs = []

    @torch.no_grad()
    def ver_test(self, backbone: torch.nn.Module, global_step: int, epoch: int ):
        results = {}
        thresh_results = {}
        self.ver_accs = []
        for i in range(len(self.ver_list)):
            tag = os.path.basename(self.ver_name_list[i])
            ## verification accuracy test
            acc, thresh, tar,far = verification.test_image_dataloader_with_fold(
                self.ver_list[i], backbone)
            logging.info('[%s][%d]' % (self.ver_name_list[i], global_step))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f, Threshold: %1.3f' % (self.ver_name_list[i], global_step, acc, thresh))

            ## write to tensorboard - per dataset
            self.summary_writer: SummaryWriter
            name = self.ver_name_list[i]
            self.summary_writer.add_scalar(f"Accuracy/{tag}", acc, epoch)
            self.summary_writer.add_scalar(f"Threshold/{tag}", thresh, epoch)

            thresh_results[tag] = thresh
            if acc > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (name, global_step, self.highest_acc_list[i]))
            results[tag] = acc
            self.ver_accs.append(acc)
    
    def triplet_loss_test(self, backbone: torch.nn.Module, global_step: int, epoch: int):
        self.ver_losses = []
        results = {}
        ## computing triplet loss test 
        for i in range(len(self.ver_list)):
            triplet_loss = verification.test_triplet_loss(backbone=backbone, dataloader=self.ver_list[i])

            ## logging
            name = self.ver_name_list[i]
            tag = os.path.basename(name)
            logging.info('[%s][%d]Triplet Loss: %1.5f' % (name, global_step, triplet_loss))
            self.ver_losses.append(triplet_loss)
            results[tag] = triplet_loss
            

        #write to tensorboard
        self.summary_writer.add_scalars("Triplet Loss", results, epoch)


    def init_dataset(self, val_targets, data_dir, image_size):
        for path in val_targets:
            # path = os.path.join(data_dir, name + ".bin")
            # path = os.path.join(data_dir, name)
            if os.path.exists(path):
                subset_num_ids = None
                norm_path = os.path.normpath(path)
                if self.subset_train_num_ids is not None and norm_path.endswith("train"):
                    subset_num_ids = self.subset_train_num_ids
                # data_set = verification.load_image_folder(path, image_size) #set dataset as dataloader
                data_loader = verification.load_image_folder(
                    path,
                    image_size,
                    subset_num_ids=subset_num_ids,
                    subset_seed=self.subset_seed,
                ) #set dataset as dataloader

                self.ver_list.append(data_loader)
                self.ver_name_list.append(path)

    def __call__(self, num_update, backbone: torch.nn.Module, epoch: int):
        if self.rank is 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update, epoch)
            # self.triplet_loss_test(backbone, num_update, epoch)
            backbone.train()
            return self.ver_accs #, self.ver_losses


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0,writer=None):
        self.frequent: int = frequent
        self.rank: int = 0 #distributed.get_rank()
        self.world_size: int = 1 #distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                #time_now = (time.time() - self.time_start) / 3600
                #time_total = time_now / ((global_step + 1) / self.total_step)
                #time_for_end = time_total - time_now
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec/3600
                if self.writer is not None:
                    # write step level
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                    #write epoch level 
                    self.writer.add_scalar('epoch learning_rate', learning_rate, epoch)
                    self.writer.add_scalar('epoch loss', loss.avg, epoch)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()
