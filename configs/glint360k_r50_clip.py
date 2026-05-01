import os
from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = True
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 128
config.lr = 0.01
config.verbose = 2000
config.dali = False
config.optimizer = "adamw"

config.rec = "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_fullface/train"
config.val_targets = ["/media/yoav/Yoav/datasets/glint360k/imageFolder_split_fullface/val", "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_narrow_eyes/val" ]
config.train_targets = ["/media/yoav/Yoav/datasets/glint360k/imageFolder_split_fullface/train", "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_narrow_eyes/train" ]
config.fullface_dir = "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_fullface/train"
config.partial_face_dir = "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_narrow_eyes/train"
config.num_classes = len(os.listdir(config.partial_face_dir))
config.num_image = sum([len(os.listdir(os.path.join(config.partial_face_dir, cls))) for cls in os.listdir(config.partial_face_dir)])
config.num_epoch = 20
config.warmup_epoch = 4

config.teacher_network = 'r50'
config.teacher_model_path = 'work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt'

config.temperature = 0.07