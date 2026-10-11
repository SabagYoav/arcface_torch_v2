from easydict import EasyDict as edict

# Fine-tune the R50 teacher (16backbone.pth) with ArcFace on a MIX of full faces
# and on-the-fly ROI-0.15 crops (train_v5_pairs.py). Same 36k-id / 10-per-id
# data, batch size, epochs and warmup as the swin_base distillation runs in
# work_dirs/exp_loss_ablation and exp_arch_comparison_10perid, so results are
# comparable; evaluated with the same LFW ROI-0.15 protocol as
# zero_shot_teacher_baseline.py (one model embeds both partial and full).
config = edict()
config.margin_list = (1.0, 0.5, 0.0)  # ArcFace: cos(theta + 0.5)
config.network = "r50"
config.pretrained_backbone_path = "16backbone.pth"
config.resume = True
config.save_all_states = True
config.output = "work_dirs/exp_finetune_r50_arcface/r50_arcface_mixed_roi15"
config.embedding_size = 512
config.sample_rate = 1.0
config.interclass_filtering_threshold = 0
config.fp16 = True
config.batch_size = 448
config.optimizer = "adamw"
config.lr = 1e-3               # PartialFC head (randomly initialized)
config.backbone_lr_mult = 0.1  # pretrained backbone trains at 1e-4
config.weight_decay = 0.1
config.frequent = 50
config.dali = False
config.dali_aug = False
config.gradient_acc = 1
config.seed = 2048
config.num_workers = 8
config.num_epoch = 10
config.warmup_epoch = 2
config.batch_grid_tag = "mixed_roi15"

config.use_onthefly_mixed_dataset = True
config.p_roi = 0.5
config.roi_ratio = 0.15
config.root_dir = "/DATA/glint360k_arch_comparison/train"
config.eye_center_metadata_path = "/DATA/glint360k_arch_comparison/train_eye_center_metadata_10perid.json"
config.num_classes = 35998
config.num_image = 334708

config.use_roi_verification = True
config.glint_val_root = "/DATA/glint360k_arch_comparison/val"
config.glint_val_metadata_path = "/DATA/glint360k_arch_comparison/val_eye_center_metadata.json"
