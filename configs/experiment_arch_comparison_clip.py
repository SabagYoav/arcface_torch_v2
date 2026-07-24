from easydict import EasyDict as edict

# Architecture comparison: distill several student backbones (r50, r100, vit_b16,
# swin_tiny, mobilevit_s) from the R50 teacher (16backbone.pth) via train_v4_clip.py,
# across ROI ratios [0.15, 0.2, 0.4, 0.6, 0.8, 1.0], on a 36k-train/9k-val identity
# full-face glint360k subset (see data_scratches/sample_arch_comparison_subset.py)
# — separate from the main 360k-identity ROI sweep. training_arch_comparison.py
# (the orchestrator) overrides network / batch_size / roi_ratio / output /
# num_classes / num_image / eye_center_metadata_path / glint_val_* per run.

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"           # overridden per architecture by the orchestrator
config.resume = True
config.save_all_states = True
config.output = None
config.embedding_size = 512
config.sample_rate = 3.0
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 256           # overridden per architecture by the orchestrator
config.lr = 0.0003
config.verbose = 2000
config.frequent = 50
config.dali = False
config.dali_aug = False
config.optimizer = "adamw"

config.num_epoch = 10             # shorter than the main sweep's 20 -- exploratory comparison, smaller subset
config.warmup_epoch = 2
config.num_workers = 2

config.teacher_network = "r50"
config.teacher_model_path = "16backbone.pth"
config.temperature = 0.07
