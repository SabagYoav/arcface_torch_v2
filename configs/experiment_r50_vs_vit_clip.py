import os
from easydict import EasyDict as edict

# Experiment: ViT-S student vs R50 teacher CLIP distillation across ROI ratios.
# ViT-S student, R50 teacher. Hyperparameters here are TUNED FOR ViT (lr, epochs,
# warmup) rather than copied from the R50 config, so this is a "best-effort per
# architecture" comparison, NOT an identical-hyperparameter one. Shared with the
# R50 run: network-agnostic pieces (embedding_size, temperature, teacher, data,
# batch_size=64 so CLIP in-batch negatives match). training_multi_loops.py (the
# orchestrator) overrides use_onthefly_dataset / root_dir / eye_center_metadata_path
# / roi_ratio / output / num_classes / num_image per ROI ratio in the sweep;
# validation runs against LFW (see utils/clip_verifications_utils.py), not a
# held-out glint360k split.

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
config.batch_size = 560                  # ~12.2GB@128, ~32.9GB@384 -> extrapolated to use most of the 49GB A6000
config.lr = 0.0003                       # ViT-tuned: 1e-3 is unstable for ViT at batch 64; 3e-4 + long warmup is stable
config.verbose = 2000
config.frequent = 50
config.dali = False
config.dali_aug = False
config.optimizer = "adamw"

config.num_epoch = 20                    # ViT from scratch needs a long schedule to converge (3 was far too few)
config.warmup_epoch = 4                  # longer warmup stabilizes early ViT training
config.num_workers = 2                   # fewer CPU workers => less heat (helps avoid the thermal shutdown)

config.teacher_network = "r50"           # teacher (same as r50-vs-r50)
config.teacher_model_path = "16backbone.pth"

config.temperature = 0.07
