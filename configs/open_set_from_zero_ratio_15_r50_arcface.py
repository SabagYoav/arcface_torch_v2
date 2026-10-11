import os
from pathlib import Path

from easydict import EasyDict as edict

# "From-zero" open-set experiment: reproduces exp_glint360k_roi_15_r50_arcface.py's
# hyperparameters, but pointed at the restored 36K/9K identity subset
# (/DATA/glint360k_arch_comparison) since the original /datasets/glint360k/ROIs/
# data no longer exists on this machine. Val identities (9K) are genuinely unseen
# during training -- a real open-set probe set, not just a held-out split of the
# training identities.
#
# Uses the fixed eval/verification.py test_fold (calibration/report identity
# split -- see that file's docstring) instead of the old same-data leak.

config = edict()
config.margin_list = (1.0, 0.5, 0.0)  # ArcFace, matching exp_glint360k_roi_15_r50_arcface.py
config.network = "r50"
config.resume = True
config.save_all_states = True
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 5e-4
config.batch_size = 256
config.lr = 0.001
config.verbose = 2000
config.frequent = 50
config.dali = False

config.rec = "/DATA/glint360k_arch_comparison/ROIs/ratio_15/train"
config.val_targets = [
    "/DATA/glint360k_arch_comparison/ROIs/ratio_15/val",
    "/DATA/glint360k_arch_comparison/ROIs/ratio_15/train",
]
# If train is listed in val_targets, evaluate on a small sampled subset of train IDs.
config.eval_subset_train_num_ids = 1000
config.eval_subset_seed = 42
config.num_classes = len(os.listdir(config.rec))
config.num_image = sum(
    sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for cls_dir in Path(config.rec).iterdir()
    if cls_dir.is_dir()
)
config.num_epoch = 20
config.warmup_epoch = 2

config.output = None
config.num_workers = 2
