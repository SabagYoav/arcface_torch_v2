from easydict import EasyDict as edict

# Fine-tune the R50 teacher (16backbone.pth) with ArcFace on a MIX of full faces
# and on-the-fly ROI-0.15 crops over the ENTIRE glint360k (train_v5_pairs.py,
# GPUs 0-2 via torchrun -- see scripts_run/glint_full_finetune_pipeline.sh).
# Evaluated on LFW only: partial(ROI-0.15)-vs-full and full-vs-full, with the
# fine-tuned model embedding both sides (same protocol as
# zero_shot_teacher_baseline.py). No glint val split: every glint identity is
# in training.
config = edict()
config.margin_list = (1.0, 0.5, 0.0)  # ArcFace: cos(theta + 0.5)
config.network = "r50"
config.pretrained_backbone_path = "16backbone.pth"
config.resume = True
config.save_all_states = True
config.output = "work_dirs/exp_finetune_r50_arcface/r50_arcface_mixed_roi15_glint_full"
config.embedding_size = 512
config.sample_rate = 1.0       # 360k classes sharded over 3 GPUs (~120k each)
config.interclass_filtering_threshold = 0
config.fp16 = True
config.batch_size = 448        # per GPU -> 1344 global (3 GPUs)
config.optimizer = "adamw"
config.lr = 1e-3               # PartialFC head (randomly initialized)
config.backbone_lr_mult = 0.1  # pretrained backbone trains at 1e-4
config.weight_decay = 0.1
config.frequent = 200
config.dali = False
config.dali_aug = False
config.gradient_acc = 1
config.seed = 2048
config.num_workers = 12
config.num_epoch = 2           # ~17M images/epoch; each epoch is ~10x the ROI views of the 36k-id run
config.warmup_epoch = 0.2
config.batch_grid_tag = "mixed_roi15"

config.use_onthefly_mixed_dataset = True
config.p_roi = 0.5
config.roi_ratio = 0.15
config.root_dir = "/DATA/glint360k_download/glint360k_images"
config.eye_center_metadata_path = "/DATA/glint360k_download/eye_center_metadata/center_eyes_metadata.json"
config.num_classes = None      # filled from the dataset (detected-only identities)
config.num_image = None        # filled from the dataset (detected-only images)

config.use_roi_verification = True
