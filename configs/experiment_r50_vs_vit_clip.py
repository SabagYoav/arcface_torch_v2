import os
from easydict import EasyDict as edict

# Experiment: ViT-S student vs R50 teacher CLIP distillation across ROI ratios.
# ViT-S student, R50 teacher. Hyperparameters here are TUNED FOR ViT (lr, epochs,
# warmup) rather than copied from the R50 config, so this is a "best-effort per
# architecture" comparison, NOT an identical-hyperparameter one. Shared with the
# R50 run: network-agnostic pieces (embedding_size, temperature, teacher, data,
# batch_size=64 so CLIP in-batch negatives match). The orchestrator overrides
# root_ff / root_pf / val_targets / train_targets / output / num_classes / num_image.

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_s_dp005_mask_0"    # student (the only change vs r50-vs-r50)
config.resume = True
config.save_all_states = True
config.output = None
config.embedding_size = 512
config.sample_rate = 3.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 64                   # fits ViT-S + R50 teacher in ~12GB; matches R50 run's CLIP negatives
config.lr = 0.0003                       # ViT-tuned: 1e-3 is unstable for ViT at batch 64; 3e-4 + long warmup is stable
config.verbose = 2000
config.frequent = 50
config.dali = False
config.dali_aug = False
config.optimizer = "adamw"

# Default (overridden by the orchestrator). Points at the quick-run subset.
ff_dir = "/media/yoav/Yoav/datasets/glint360k/fullface_subset"
pf_dir = "/media/yoav/Yoav/datasets/variants_dataset_subset/ratio_100"

config.root_ff = f"{ff_dir}/train"
config.root_pf = f"{pf_dir}/train"
config.val_targets = [f"{ff_dir}/val", f"{pf_dir}/val"]
config.train_targets = [f"{ff_dir}/train", f"{pf_dir}/train"]

config.num_epoch = 20                    # ViT from scratch needs a long schedule to converge (3 was far too few)
config.warmup_epoch = 4                  # longer warmup stabilizes early ViT training
config.num_workers = 2                   # fewer CPU workers => less heat (helps avoid the thermal shutdown)

config.teacher_network = "r50"           # teacher (same as r50-vs-r50)
config.teacher_model_path = "work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt"

config.temperature = 0.07
