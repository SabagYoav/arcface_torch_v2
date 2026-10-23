from easydict import EasyDict as edict

# Extend work_dirs/exp_clip_r50_vs_swin_glint/clip_ratio_15 (swin_base, CLIP
# distillation from the r50 teacher, full glint360k, ROI 0.15) from 20 to 30
# epochs. Resumes from a COPY of its epoch-20 checkpoint in a new output dir;
# the original run's folder is never written to. Every setting below is copied
# from that run's training.log except num_epoch and output.
config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "swin_base"
config.resume = True
config.save_all_states = True
config.output = "work_dirs/exp_clip_r50_vs_swin_glint/clip_ratio_15_ext30"
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
config.num_workers = 2
config.num_epoch = 30          # was 20
config.warmup_epoch = 4
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
