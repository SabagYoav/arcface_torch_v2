from easydict import EasyDict as edict

# iResNet-50 student, CLIP (InfoNCE) distillation from the R50 teacher on the
# full glint360k at ROI 0.15, 10 epochs, 1 GPU. Same as the swin_base run
# work_dirs/exp_clip_r50_vs_swin_glint/clip_ratio_15 (same teacher, data,
# batch, lr, temperature, seed) except config.network, output, and the
# schedule length: 10 epochs (swin: 20) with warmup scaled to keep 20% (2 vs 4).
config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = True
config.save_all_states = True
config.output = "work_dirs/exp_clip_r50_vs_r50_glint/clip_ratio_15"
config.embedding_size = 512
config.sample_rate = 3.0
config.fp16 = True
config.batch_size = 576
config.optimizer = "adamw"
config.lr = 0.0003
config.weight_decay = 0.1
config.verbose = 2000
config.frequent = 50
config.dali = False
config.dali_aug = False
config.gradient_acc = 1
config.seed = 2048
config.num_workers = 8            # swin run used 2; loader-only, no effect on results (no random aug)
config.num_epoch = 10
config.warmup_epoch = 2
config.teacher_network = "r50"
config.teacher_model_path = "16backbone.pth"
config.temperature = 0.07
config.use_onthefly_dataset = True
config.root_dir = "/DATA/glint360k_download/glint360k_images"
config.eye_center_metadata_path = "/DATA/glint360k_download/eye_center_metadata/center_eyes_metadata.json"
config.roi_ratio = 0.15
config.batch_grid_tag = "clip_ratio_15"
config.num_classes = 360187
config.num_image = 17086125
