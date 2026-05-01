from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = True
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
# config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256 
config.lr = 0.001 #0.001
config.verbose = 2000
config.dali = False

#################
### experiment specific setting ###
##################
datase_name = "glint360k"
roi = "narrow_eyes"
title = f"dev_test_overfit"
###################

config.rec = f"/datasets/{datase_name}/imageFolder_split_{roi}/train"
config.val_targets = [f'/datasets/{datase_name}/imageFolder_split_{roi}/val',f"/datasets/{datase_name}/imageFolder_split_{roi}/train" ] #['/home/yoav/DATA/val_sets/lfw.bin', '/home/yoav/DATA/val_sets/cfp_fp.bin']
config.num_classes = 63013
config.num_image = 1446728
config.num_epoch = 20
config.warmup_epoch = 2

# config.output = f'work_dirs/{datase_name}_{roi}_augmentations_lr{config.lr}_batch{config.batch_size}_{title}'#'work_dirs/glint360k_paired_narrow_fullface'
config.output = None
# dataload numworkers
config.num_workers = 4
