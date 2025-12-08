from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 32
config.lr = 0.001
config.verbose = 40
config.dali = False

config.rec = "/Downloads/Yoav/datasets/glint360k/imageFolder_split_fullface/train"
config.val_targets = ['/Downloads/Yoav/datasets/glint360k/imageFolder_split_fullface/test', "/Downloads/Yoav/datasets/glint360k/imageFolder_split_fullface/train"]

config.num_classes = 10
config.num_image = 100
config.num_epoch = 2000
config.warmup_epoch = 0
#['lfw', 'cfp_fp', "agedb_30"]
