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
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

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
    df_10 = df[(df["yaw"].abs() > 5) & (df["yaw"].abs() <= 10)].reset_index(drop=True)
    df_20 = df[(df["yaw"].abs() > 10) & (df["yaw"].abs() <= 20)].reset_index(drop=True)

    print(f"|yaw| <= 5: {len(df_5)}, |yaw| <= 10: {len(df_10)}, |yaw| <= 20: {len(df_20)}, total: {len(df)}")  
    return df_5, df_10, df_20, df

def plot_yaw_distribution(df_5, df_10, df_20, prefix=""):
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for df, deg, color in zip([df_5, df_10, df_20], [5, 10, 20], colors):
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
    common_ids = (get_ids(df_5) & get_ids(df_10) & get_ids(df_20))
    print("Common IDs:", len(common_ids))

    df_5_eq  = filter_by_ids(df_5,  common_ids)
    df_10_eq = filter_by_ids(df_10, common_ids)
    df_20_eq = filter_by_ids(df_20, common_ids)

    global K
    K = 100 # very common in verification

    df_5_eq  = sample_k_per_id(df_5_eq,  K)
    df_10_eq = sample_k_per_id(df_10_eq, K)
    df_20_eq = sample_k_per_id(df_20_eq, K)
    
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



if __name__ == "__main__":
    ## init args
    default_testset_root = "datasets/glint360k/imageFolder_split_fullface/test"
    json_path = "data_analytics/eyes_test_set_pose_results.json"
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_root", default=default_testset_root, type=str)
    parser.add_argument("--json_path", default=json_path, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()

    ## prepare data as dfs
    df_5, df_10, df_20, df = split_by_yaw_ranges(args.json_path)
    plot_yaw_distribution(df_5, df_10, df_20, prefix="before_equalization_")
    df_5, df_10, df_20 = equalize_same_ids(df_5, df_10, df_20)
    plot_yaw_distribution(df_5, df_10, df_20, prefix="after_equalization_")

    df_5  = limit_to_n_ids(df_5,  2500)
    df_10 = limit_to_n_ids(df_10, 2500)
    df_20 = limit_to_n_ids(df_20, 2500)

    plot_yaw_distribution(df_5, df_10, df_20, prefix="limited_3K_ids_")

    ## load model same as training loop
    backbone = load_backbone('work_dirs/checkpoints/augmentation/arcface/fullface/model_best.pt')

    ## prepare data loaders same as training loop
    loader_5  = get_loader(df_5, batch_size=args.batch_size)
    loader_10 = get_loader(df_10, batch_size=args.batch_size)
    loader_20 = get_loader(df_20, batch_size=args.batch_size)

    loader_5_n_10 = get_cross_pose_pair_loader(df_pose_a = df_5,df_pose_b = df_10, batch_size=args.batch_size)
    loader_5_n_20 = get_cross_pose_pair_loader(df_pose_a = df_5,df_pose_b = df_20, batch_size=args.batch_size)
    loader_10_n_20 = get_cross_pose_pair_loader(df_pose_a = df_10,df_pose_b = df_20, batch_size=args.batch_size)

    acc_5,  best_thr = test_image_dataloader_with_fold(loader_5, backbone)
    print(f"|yaw| ≤ 5° vs  |yaw| ≤ 5° : {acc_5:.4f}, best_thr: {best_thr:.4f}")
    acc_10, best_thr = test_image_dataloader_with_fold(loader_10, backbone)
    print(f"|yaw| ≤ 10° vs  |yaw| ≤ 10° : {acc_10:.4f}, best_thr: {best_thr:.4f}")
    acc_20, best_thr = test_image_dataloader_with_fold(loader_20, backbone)
    print(f"|yaw| ≤ 20° vs  |yaw| ≤ 20° : {acc_20:.4f}, best_thr: {best_thr:.4f}")

    acc_10 = test_cross_pose(loader_5_n_10, backbone, threshold=0.8)
    print(f"|yaw| ≤ 10° vs  |yaw| ≤ 5° : {acc_10:.4f}")
    acc_20 = test_cross_pose(loader_5_n_20, backbone, threshold=0.8)
    print(f"|yaw| ≤ 20° vs  |yaw| ≤ 5° : {acc_20:.4f}")
    acc_20 = test_cross_pose(loader_10_n_20, backbone, threshold=0.8)
    print(f"|yaw| ≤ 20° vs  |yaw| ≤ 10° : {acc_20:.4f}")
