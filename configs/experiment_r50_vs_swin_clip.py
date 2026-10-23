import os
from easydict import EasyDict as edict

# Experiment: Swin-Base student vs R50 teacher CLIP distillation across ROI
# ratios, on the full glint360k dataset. Swin-Base won the smaller-subset
# architecture comparison (see training_arch_comparison.py /
# work_dirs/exp_arch_comparison_10perid), so it replaces the ViT-S student
# used in the original full-dataset sweep (configs/experiment_r50_vs_vit_clip.py,
# EXP_NAME exp_clip_r50_vs_vit_full — left untouched). training_multi_loops.py
# (the orchestrator) overrides use_onthefly_dataset / root_dir /
# eye_center_metadata_path / roi_ratio / output / num_classes / num_image per
# ROI ratio in the sweep; validation runs against LFW (see
# utils/clip_verifications_utils.py), not a held-out glint360k split.

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "swin_base"
config.resume = True
config.save_all_states = True
config.output = None
config.embedding_size = 512
config.sample_rate = 3.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 576                  # measured: ~41.9GB synthetic @576 on a full 49GB A6000, comfortable margin
config.lr = 0.0003
config.verbose = 2000
config.frequent = 50
config.dali = False
config.dali_aug = False
config.optimizer = "adamw"

config.num_epoch = 20
config.warmup_epoch = 4
config.num_workers = 2

config.teacher_network = "r50"
config.teacher_model_path = "16backbone.pth"

config.temperature = 0.07
