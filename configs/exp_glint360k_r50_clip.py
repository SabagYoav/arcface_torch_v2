

# ff_dir = "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_fullface"
# pf_dir = "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_narrow_eyes"

# Auto-generated config file
from easydict import EasyDict as edict

config = edict()
config.network = "r50" #'vit_s_dp005_mask_0'
config.embedding_size = 512
config.fp16 = True
config.batch_size = 64
config.optimizer = 'adamw'
config.lr = 0.001
config.weight_decay = 0.1
config.num_epoch = 20
config.warmup_epoch = 2
config.teacher_network = 'r50'
config.teacher_model_path = 'work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt'
config.temperature = 0.07


config.root_ff = '/datasets/glint360k/imageFolder_split_fullface/train'
config.root_pf = '/datasets/variants_dataset/train'
config.val_targets = ['/datasets/glint360k/imageFolder_split_fullface/val', '/datasets/variants_dataset/val']
config.train_targets = ['/datasets/glint360k/imageFolder_split_fullface/train', '/datasets/variants_dataset/train']
config.num_classes = 63013
config.num_image = 1446728

config.resume = False
config.save_all_states = True
config.seed = 2048

config.output = None
config.batch_grid_tag = 'clip'



config.sample_rate = 3.0

config.verbose = 2000
config.frequent = 10
config.dali = False
config.dali_aug = False
config.gradient_acc = 1
config.num_workers = 0
config.wandb_key = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
config.suffix_run_name = None
config.using_wandb = False
config.wandb_entity = 'entity'
config.wandb_project = 'project'
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False
