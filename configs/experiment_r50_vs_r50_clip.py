import os
from easydict import EasyDict as edict

# Experiment: R50 student vs R50 teacher CLIP distillation across ROI ratios.
# Base config consumed by training_multi_loops.py; the orchestrator overrides
# root_ff / root_pf / val_targets / train_targets / output / num_classes / num_image
# per ROI ratio.

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"                   # student
config.resume = True                      # resume from checkpoint_gpu_0.pt in the ratio's output dir if present
config.save_all_states = True             # save per-epoch checkpoint (backbone+opt+sched) so a ratio can resume
config.output = None
config.embedding_size = 512
config.sample_rate = 3.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128                    # keep student+teacher R50 under ~0.5 of a 12GB GPU
config.lr = 0.001
config.verbose = 2000
config.frequent = 50
config.dali = False
config.dali_aug = False
config.optimizer = "adamw"

# Default (overridden by the orchestrator). Points at the quick-run subset.
ff_dir = "/media/yoav/Yoav/datasets/variants_dataset_subset/ratio_100"
pf_dir = "/media/yoav/Yoav/datasets/variants_dataset_subset/ratio_100"

config.root_ff = f"{ff_dir}/train"
config.root_pf = f"{pf_dir}/train"
config.val_targets = [f"{ff_dir}/val", f"{pf_dir}/val"]
config.train_targets = [f"{ff_dir}/train", f"{pf_dir}/train"]

config.num_epoch = 5
config.warmup_epoch = 2
config.num_workers = 4

config.teacher_network = "r50"           # teacher
config.teacher_model_path = "work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt"

config.temperature = 0.07
