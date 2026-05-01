import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils.utils_config import get_config
from dataset import get_dataloader
from backbones import get_model
from eval.pose_datasets import get_cross_pose_pair_loader, get_loader
from eval.verification import test_image_dataloader, test_image_dataloader_with_fold

def split_by_yaw_ranges(json_path):
    """
    Load pose results JSON and split into yaw-range DataFrames.

    Returns:
        df_5  : |yaw| <= 5
        df_10 : |yaw| <= 10
        df_20 : |yaw| <= 20
    """
    df = pd.read_json(json_path)

    df_5  = df[df["yaw"].abs() <= 5].reset_index(drop=True)
    df_5_10 = df[(df["yaw"].abs() > 5) & (df["yaw"].abs() <= 10)].reset_index(drop=True)
    df_10_20 = df[(df["yaw"].abs() > 10) & (df["yaw"].abs() <= 20)].reset_index(drop=True)
    df_30_90 = df[(df["yaw"].abs() > 30)].reset_index(drop=True)

    print(f"|yaw| <= 5: {len(df_5)}, |yaw| <= 10: {len(df_5_10)}, |yaw| <= 20: {len(df_10_20)}, |yaw| <= 30: {len(df_30_90)}, total: {len(df)}")  
    return df_5, df_5_10, df_10_20, df_30_90, df

def plot_yaw_distribution(df_5, df_10, df_20, df_30, prefix=""):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for df, deg, color in zip([df_5, df_10, df_20, df_30], [5, 10, 20, 30], colors):
        plt.hist(df["yaw"], bins=30, edgecolor='black', color=color)
        plt.xlabel("Yaw (degrees)")
        plt.ylabel("Count")
        plt.title(f"Yaw Distribution (|yaw| ≤ {deg}°)")
        plt.savefig(f"{prefix}yaw_distribution_{deg}deg.png")
        plt.close()

def sample_k_per_id(df, k, seed=42):
    random.seed(seed)
    out = []
    for pid, g in df.groupby(df["path"].apply(lambda p: Path(p).parent.name)):
        # Sample up to k per id, or all if less than k
        if len(g) <= k:
            sampled = g
        sampled = g.sample(min(k, len(g)), random_state=seed)
        out.append(sampled)
    if out:
        return pd.concat(out).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=df.columns)

def get_ids(df):
    return set(df["path"].apply(lambda p: Path(p).parent.name))

def filter_by_ids(df, valid_ids):
    return df[
        df["path"].apply(lambda p: Path(p).parent.name in valid_ids)
    ].reset_index(drop=True)

def extract_id(path):
    return Path(path).parent.name

import random

def limit_to_n_ids(df, n_ids=1000, seed=42):
    all_ids = sorted(df["path"].apply(extract_id).unique())
    assert len(all_ids) >= n_ids, "Not enough IDs in dataframe"

    random.seed(seed)
    selected_ids = set(random.sample(all_ids, n_ids))

    return df[df["path"].apply(extract_id).isin(selected_ids)].reset_index(drop=True)


def equalize_same_ids(df_5, df_10, df_20):
    """
    Equalize the number of samples in each DataFrame to the smallest one.
    """
    common_ids = (get_ids(df_5) & get_ids(df_10) & get_ids(df_20) )
    print("Common IDs:", len(common_ids))

    df_5_eq  = filter_by_ids(df_5,  common_ids)
    df_10_eq = filter_by_ids(df_10, common_ids)
    df_20_eq = filter_by_ids(df_20, common_ids)
    # df_30_eq = filter_by_ids(df_30, common_ids)

    global K
    K = 100 # very common in verification

    df_5_eq  = sample_k_per_id(df_5_eq,  K)
    df_10_eq = sample_k_per_id(df_10_eq, K)
    df_20_eq = sample_k_per_id(df_20_eq, K)
    # df_30_eq = sample_k_per_id(df_30_eq, K)
    
    return df_5_eq, df_10_eq, df_20_eq



def verification_accuracy(pos_scores, neg_scores, threshold):
    tp = (pos_scores >= threshold).sum().item()
    fn = (pos_scores <  threshold).sum().item()
    tn = (neg_scores <  threshold).sum().item()
    fp = (neg_scores >= threshold).sum().item()

    acc = (tp + tn) / (tp + tn + fp + fn)
    return acc

def balanced_accuracy(TP, TN, FP, FN):
    TPR = TP / (TP + FN + 1e-8)   # recall / sensitivity
    TNR = TN / (TN + FP + 1e-8)   # specificity
    return 0.5 * (TPR + TNR)

def weighted_accuracy(TP, TN, FP, FN):
    P = TP + FN
    N = TN + FP

    w_pos = 0.5 / P
    w_neg = 0.5 / N

    return w_pos * TP + w_neg * TN

def valnila_accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

import collections
def load_backbone(model_path):
    backbone = get_model('r50', dropout=0.0, fp16=True, num_features=512).cuda()
    loaded = torch.load(model_path)

    if isinstance(loaded, dict) and "state_dict_backbone" in loaded:
        backbone.load_state_dict(loaded["state_dict_backbone"])
    elif isinstance(loaded, collections.OrderedDict):
        backbone.load_state_dict(loaded)
    else:
        raise ValueError("Unknown checkpoint format")

    backbone.eval()
    return backbone

import matplotlib.pyplot as plt

def save_score_hist(pos_scores, neg_scores, out_path, title=None):
    plt.figure(figsize=(6, 4))

    plt.hist(pos_scores.cpu().numpy(), bins=50, alpha=0.5, label="pos")
    plt.hist(neg_scores.cpu().numpy(), bins=50, alpha=0.5, label="neg")

    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.legend()
    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def test_cross_pose(loader, backbone, threshold):
    backbone.eval()

    TP = TN = FP = FN = 0

    with torch.no_grad():
        for img_a, img_b_pos, img_b_neg, labels in loader:

            emb_a = F.normalize(backbone(img_a.cuda()), dim=1)
            emb_pos = F.normalize(backbone(img_b_pos.cuda()), dim=1)
            emb_neg = F.normalize(backbone(img_b_neg.cuda()), dim=1)

            # cosine similarities
            pos_scores = (emb_a * emb_pos).sum(dim=1)
            neg_scores = (emb_a * emb_neg).sum(dim=1)

            # element-wise decisions
            TP += (pos_scores > threshold).sum().item()
            FN += (pos_scores <= threshold).sum().item()

            TN += (neg_scores < threshold).sum().item()
            FP += (neg_scores >= threshold).sum().item()
        
            save_score_hist(pos_scores, neg_scores, out_path="cross_pose_score_hist.png", title="Cross-Pose Score Distribution")

    acc = valnila_accuracy(TP, TN, FP, FN)
    return acc            

def save_df_in_temp_dir(df, save_path="/media/temp_eval_dataset/", src_root_prefix=Path("/datasets/glint360k/ROIs/ratio_20/")): #TODO: remove hardcoded paths
    temp_dir = Path(save_path)
    # only if not exists
    # if not temp_dir.exists():
    temp_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in df.iterrows():
        img_path = src_root_prefix / Path(row["path"])
        id_name = img_path.parent.name
        dest_dir = temp_dir / id_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / img_path.name
        if not dest_path.exists():
            # copy file
            with open(img_path, "rb") as f_src:
                with open(dest_path, "wb") as f_dst:
                    f_dst.write(f_src.read())


def get_loader_from_df(df: pd.DataFrame, cfg, save_path:str): #TODO: check args type
    save_df_in_temp_dir(df, save_path)
    # Remove empty class directories that would cause ImageFolder to fail
    for d in Path(save_path).iterdir():
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    dataloader = get_dataloader(save_path, local_rank=0, batch_size=cfg.batch_size,  seed=cfg.seed, num_workers=cfg.num_workers )
    return dataloader

class CrossPoseSampler:
    def __init__(self, dataloader_a, dataloader_b):
        self.dataloader_a = dataloader_a
        self.dataloader_b = dataloader_b
        self.iter_a = iter(self.dataloader_a)
        self.iter_b = iter(self.dataloader_b)

    def sample_crosspose(self):
        batch_a = next(self.iter_a)
        batch_b = next(self.iter_b)

        # Assuming each batch is (images, labels)
        if isinstance(batch_a, (list, tuple)) and len(batch_a) >= 2:
            img_a, labels_a = batch_a[0], batch_a[1]
        else:
            img_a, labels_a = batch_a, None

        if isinstance(batch_b, (list, tuple)) and len(batch_b) >= 2:
            img_b, labels_b = batch_b[0], batch_b[1]
        else:
            img_b, labels_b = batch_b, None

        return img_a, img_b, labels_a, labels_b


@torch.no_grad()
def batch_cross_accuracy(emb_a, lab_a, emb_b, lab_b, thresholds):
    """
    emb_a: [Ba, D]
    emb_b: [Bb, D]
    lab_a: [Ba]
    lab_b: [Bb]
    """

    S = emb_a @ emb_b.T   # [Ba, Bb]

    same = lab_a.unsqueeze(1) == lab_b.unsqueeze(0)

    pos = S[same]
    neg = S[~same]

    if len(pos) == 0 or len(neg) == 0:
        return None, None  # skip invalid batch

    w_pos = 1.0
    w_neg = len(pos) / len(neg)

    best_acc, best_acc_thr = 0.0, None
    best_tar, best_tar_thr = 0.0, None
    best_far, best_far_thr = 1.0, None

    for t in thresholds:
        tp = (pos > t).sum().float()
        fn = (pos <= t).sum().float()
        tn = (neg <= t).sum().float()
        fp = (neg > t).sum().float()

        acc = (w_pos * tp + w_neg * tn) / (
            w_pos * (tp + fn) + w_neg * (tn + fp) + 1e-8
        )
        tar = tp / (tp + fn + 1e-8)
        far = fp / (fp + tn + 1e-8)

        if acc > best_acc:
            best_acc = acc.item()
            best_acc_thr = t
        if tar >= best_tar :
            best_tar = tar
            best_tar_at_far = far
            best_tar_thr = t
        if far <= best_far :
            best_far = far
            best_far_thr = t
            best_far_at_tar = tar
    results = {
        "best_acc": best_acc,
        "best_acc_thr": best_acc_thr,
        "best_tar": best_tar,
        "best_tar_at_far": best_tar_at_far,
        "best_tar_thr": best_tar_thr,
        "best_far": best_far,
        "best_far_thr": best_far_thr,
        "best_far_at_tar": best_far_at_tar
    }

    return results

@torch.no_grad()
def cross_pose_test( loader_a, loader_b, backbone, device="cuda"):
    backbone.eval()

    #thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [0.2]

    accs = []
    acc_thrs = []
    tars = []
    tars_at_far = []
    tar_thrs = []
    fars = []
    fars_at_tar = []
    far_thrs = []

    for (img_a, lab_a), (img_b, lab_b) in zip(loader_a, loader_b):

        img_a = img_a.to(device)
        img_b = img_b.to(device)

        emb_a = F.normalize(backbone(img_a), dim=1)
        emb_b = F.normalize(backbone(img_b), dim=1)

        results= batch_cross_accuracy(
            emb_a.cpu(), lab_a.cpu(),
            emb_b.cpu(), lab_b.cpu(),
            thresholds
        )

        if results is not None:
            accs.append(results["best_acc"])
            acc_thrs.append(results["best_acc_thr"])
            tars.append(results["best_tar"])
            tars_at_far.append(results["best_tar_at_far"])
            tar_thrs.append(results["best_tar_thr"])
            fars.append(results["best_far"])
            fars_at_tar.append(results["best_far_at_tar"])
            far_thrs.append(results["best_far_thr"])

    mean_acc = sum(accs) / len(accs)
    mean_acc_thr = sum(acc_thrs) / len(acc_thrs)
    mean_tar = sum(tars) / len(tars)
    mean_tar_at_mean_far = sum(tars_at_far) / len(tars_at_far)   
    mean_tar_thr = sum(tar_thrs) / len(tar_thrs)
    mean_far = sum(fars) / len(fars)
    mean_far_at_mean_tar = sum(fars_at_tar) / len(fars_at_tar)
    mean_far_thr = sum(far_thrs) / len(far_thrs)

    results = {
        "mean_acc": mean_acc,
        "mean_acc_thr": mean_acc_thr,
        "mean_tar": mean_tar,
        "mean_best_tar_at_mean_far": mean_tar_at_mean_far,
        "mean_tar_thr": mean_tar_thr,
        "mean_far": mean_far,
        "mean_best_far_at_mean_tar": mean_far_at_mean_tar,
        "mean_far_thr": mean_far_thr
    }
    

    return results

if __name__ == "__main__":

    default_testset_root = "datasets/glint360k/imageFolder_split_narrow_eyes/test"
    json_path = "data_analytics/eyes_test_set_pose_results.json"
    config_fpath = "configs/exp_glint360k_roi_20_r50_arcface.py"

    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_root", default=default_testset_root, type=str)
    parser.add_argument("--json_path", default=json_path, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--config", default=config_fpath, type=str) 
    args = parser.parse_args()

    cfg = get_config(args.config)
    args.testset_root = cfg.val_targets[0]  # override testset_root with config value
    
    df_5, df_5_10, df_10_20, df_30_90, df = split_by_yaw_ranges(args.json_path)
    plot_yaw_distribution(df_5, df_5_10, df_10_20, df_30_90, prefix="before_equalization_")
    df_5, df_5_10, df_10_20  = equalize_same_ids(df_5, df_5_10, df_10_20)
    plot_yaw_distribution(df_5, df_5_10, df_10_20, df_30_90, prefix="after_equalization_")

    df_5  = limit_to_n_ids(df_5,  2500)
    df_5_10 = limit_to_n_ids(df_5_10, 2500)
    df_10_20 = limit_to_n_ids(df_10_20, 2500)
    # df_30_90 = limit_to_n_ids(df_30_90, 2500)
    plot_yaw_distribution(df_5, df_5_10, df_10_20, df_30_90, prefix="limited_3K_ids_")
    # load model
    # backbone = load_backbone('work_dirs/checkpoints/augmentation/arcface/fullface/model_best.pt')
    backbone = load_backbone("/media/yoav/Yoav/arcface_torch_v0/work_dirs/glint360k/no_augmentation/arcface/narrow_eyes/model_best.pt")

    ## load dataloaders
    loader_5  = get_loader_from_df(df_5, cfg=cfg,save_path="/media/temp_eval_dataset/pose_5/")
    loader_10 = get_loader_from_df(df_5_10, cfg=cfg,save_path="/media/temp_eval_dataset/pose_10/")
    loader_20 = get_loader_from_df(df_10_20, cfg=cfg,save_path="/media/temp_eval_dataset/pose_20/")
    # loader_90 = get_loader_from_df(df_30_90, cfg=cfg,save_path="/media/temp_eval_dataset/pose_30/")

    ## same-pose test
    same_pose_acc_5,  best_thr, tar, far = test_image_dataloader_with_fold(loader_5, backbone)
    print(f"|yaw| ≤ 5° vs  |yaw| ≤ 5° : {same_pose_acc_5:.4f}, best_thr: {best_thr:.4f}")
    print(f' same-pose TAR:: {tar:.4f} @FAR: {far:.4f}')
    same_pose_acc_10, best_thr, tar, far = test_image_dataloader_with_fold(loader_10, backbone)
    print(f"|yaw| ≤ 10° vs  |yaw| ≤ 10° : {same_pose_acc_10:.4f}, best_thr: {best_thr:.4f}")
    print(f' same-pose TAR:: {tar:.4f} @FAR: {far:.4f}')
    same_pose_acc_20, best_thr, tar, far = test_image_dataloader_with_fold(loader_20, backbone)
    print(f"|yaw| ≤ 20° vs  |yaw| ≤ 20° : {same_pose_acc_20:.4f}, best_thr: {best_thr:.4f}")
    print(f' same-pose TAR:: {tar:.4f} @FAR: {far:.4f}')
    ## cross-pose test
    cross_pose_5_10_res =  cross_pose_test(loader_5, loader_10, backbone) # TODO add FT
    print(f"|yaw| ≤ 5° vs  |yaw| ≤ 10° : {cross_pose_5_10_res['mean_acc']:.4f}, best_thr: {cross_pose_5_10_res['mean_acc_thr']:.4f}")
    print(f'cross-pose TAR:: {cross_pose_5_10_res["mean_tar"]:.4f} @FAR: {cross_pose_5_10_res["mean_best_tar_at_mean_far"]:.4f}')
    print(f'cross-pose FAR:: {cross_pose_5_10_res["mean_far"]:.4f} @TAR: {cross_pose_5_10_res["mean_best_far_at_mean_tar"]:.4f}')
    print('############')
    cross_pose_5_20_res =  cross_pose_test(loader_5, loader_20, backbone)
    print(f"|yaw| ≤ 5° vs  |yaw| ≤ 20° : {cross_pose_5_20_res['mean_acc']:.4f}, best_thr: {cross_pose_5_20_res['mean_acc_thr']:.4f}")
    print(f'cross-pose TAR:: {cross_pose_5_20_res["mean_tar"]:.4f} @FAR: {cross_pose_5_20_res["mean_best_tar_at_mean_far"]:.4f}')
    print(f'cross-pose FAR:: {cross_pose_5_20_res["mean_far"]:.4f} @TAR: {cross_pose_5_20_res["mean_best_far_at_mean_tar"]:.4f}')
    print('############')



    ## Sanity check: cross-pose for same pose
    test_pose_res_20 =  cross_pose_test(loader_20, loader_20, backbone)
    print(f"|yaw| ≤ 20° vs  |yaw| ≤ 20° (cross-pose sanity check) : {test_pose_res_20['mean_acc']:.4f}, best_thr: {test_pose_res_20['mean_acc_thr']:.4f}")
    test_pose_acc_20,  best_thr =  test_image_dataloader_with_fold(loader_20, backbone)
    print(f"|yaw| ≤ 20° vs  |yaw| ≤ 20° (same-pose sanity check) : {test_pose_acc_20:.4f}, best_thr: {best_thr:.4f}")
