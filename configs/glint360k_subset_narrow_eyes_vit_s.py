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
config.sample_rate = 0.3
config.fp16 = True
config.weight_decay = 0.1
config.batch_size = 128
config.optimizer = "adamw"
config.lr = 0.001
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

config.num_classes = 63013
config.num_image = 1446728
config.num_epoch = 40
config.warmup_epoch = config.num_epoch // 10
config.val_targets = [f'/datasets/{datase_name}/imageFolder_split_{roi}/val',f"/datasets/{datase_name}/imageFolder_split_{roi}/train" ]

config.num_workers = 4
