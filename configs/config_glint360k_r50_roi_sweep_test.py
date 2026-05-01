from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 175
config.lr = 0.01
config.verbose = 2000
config.dali = False

config.rec = f"/media/yoav/Yoav/datasets/glint360k/temp_roi_sweep_subset/train"
config.val_targets = [f"/media/yoav/Yoav/datasets/glint360k/temp_roi_sweep_subset/val",f"/media/yoav/Yoav/datasets/glint360k/temp_roi_sweep_subset/train" ]

config.num_classes = 10
config.num_image = 280
config.num_epoch = 20
config.warmup_epoch = 4
