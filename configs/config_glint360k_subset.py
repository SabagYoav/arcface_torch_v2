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
# config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256 
config.lr = 0.01#0.001
config.verbose = 2000
config.dali = False

config.rec = "/datasets/glint360k/imageFolder_split_fullface/train"
config.val_targets = ['/datasets/glint360k/imageFolder_split_fullface/val'] #['/home/yoav/DATA/val_sets/lfw.bin', '/home/yoav/DATA/val_sets/cfp_fp.bin']
config.num_classes = 63014
config.num_image = 1446728
config.num_epoch = 20
config.warmup_epoch = 5

# dataload numworkers
config.num_workers = 4
