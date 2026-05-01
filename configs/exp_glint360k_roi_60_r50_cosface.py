import os
from pathlib import Path

from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = False
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
# config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 256 
config.lr = 0.001 #0.001
config.verbose = 2000
config.dali = False

config.rec = f"/datasets/glint360k/ROIs/ratio_60/train"
config.val_targets = [f'/datasets/glint360k/ROIs/ratio_60/val', f"/datasets/glint360k/ROIs/ratio_60/train"] #['/home/yoav/DATA/val_sets/lfw.bin', '/home/yoav/DATA/val_sets/cfp_fp.bin']
# If train is listed in val_targets, evaluate on a small sampled subset of train IDs.
# Set to None to disable subsampling.
config.eval_subset_train_num_ids = 1000
config.eval_subset_seed = 42
config.num_classes = len(os.listdir(f"/datasets/glint360k/ROIs/ratio_60/train"))
config.num_image = sum(
    sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    for cls_dir in Path(f"/datasets/glint360k/ROIs/ratio_60/train").iterdir()
    if cls_dir.is_dir()
)
config.num_epoch = 20
config.warmup_epoch = 2

config.output = None

config.num_workers = 2
