import os
from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_s_dp005_mask_0"
config.resume = True
config.output = None
config.embedding_size = 512
config.sample_rate = 3.0
config.fp16 = True
# config.momentum = 0.9
config.weight_decay = 0.1
config.batch_size = 64
config.lr = 0.01
config.verbose = 2000
config.dali = False
config.optimizer = "adamw"

# root_a = "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_narrow_eyes"
ff_dir = "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_fullface"
pf_dir = "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_narrow_eyes"

config.root_ff = f"{ff_dir}/train"
config.root_pf = f"{pf_dir}/train"

config.val_targets = [f"{ff_dir}/val", f"{pf_dir}/val"]
config.train_targets = [f"{ff_dir}/train", f"{pf_dir}/train"]

# config.num_classes = len(os.listdir(config.root_pf))
# config.num_image = sum([len(os.listdir(os.path.join(config.root_pf, cls))) for cls in os.listdir(config.root_pf)])

config.num_epoch = 5
config.warmup_epoch = 2
config.num_workers = 0

config.teacher_network = 'r50'
config.teacher_model_path = 'work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt'

config.temperature = 0.07
