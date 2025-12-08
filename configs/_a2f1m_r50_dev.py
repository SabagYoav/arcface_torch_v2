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
# config.fp16 = True
config.fp16 = False
# config.momentum = 0.9
config.weight_decay = 5e-4
# config.batch_size = 1024
config.batch_size = 512
config.lr = 0.001
config.verbose = 600
config.dali = False

config.rec = "/home/yoav/DATA/arc2face_subset"
config.val_targets = ['/home/yoav/DATA/arc2face_val_subset'] #['/home/yoav/DATA/val_sets/lfw.bin', '/home/yoav/DATA/val_sets/cfp_fp.bin']
config.num_classes = 29957
config.num_image = 590163
config.num_epoch = 20
config.warmup_epoch = 5

# dataload numworkers
config.num_workers = 16
