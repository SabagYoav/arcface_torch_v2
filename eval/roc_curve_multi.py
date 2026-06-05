import gc
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# Config
# =========================
RATIOS = [15, 20, 25, 30, 40, 60, 100]
FULL_DIR = "/datasets/glint360k/ROIs/ratio_100/test"
OUT_ROC_IMG = "work_dirs/roc_partial_vs_full_multi_benchmark.png"
LOG_FILE = "work_dirs/roc_multi_progress_benchmark.json"

BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Load a single model for a given ratio
# =========================
def load_partial_model(ratio):
    from backbones import get_model
    from utils.utils_config import get_config
    config_path = f"configs/variants_config_ratio_{ratio}.py"
    cfg = get_config(config_path)
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(DEVICE)
    backbone.load_state_dict(torch.load(f"{cfg.output}/best_model.pt", map_location=DEVICE, weights_only=True))
    backbone.eval()
    return backbone, cfg


def load_fullface_model(cfg):
    from backbones import get_model
    backbone = get_model(cfg.teacher_network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(DEVICE)
    backbone.load_state_dict(torch.load(cfg.teacher_model_path, map_location=DEVICE, weights_only=True))
    backbone.eval()
    return backbone


# =========================
# Image transform
# =========================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# =========================
# Extract embeddings
# =========================
@torch.no_grad()
def extract_embeddings(model, root_dir):
    ds = datasets.ImageFolder(root=root_dir, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    all_embs = []
    all_labels = []

    for imgs, labels in tqdm(dl, desc=f"Extracting {root_dir}"):
        imgs = imgs.to(DEVICE, non_blocking=True)

        emb = model(imgs)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]

        emb = F.normalize(emb, dim=1)

        all_embs.append(emb.cpu())
        all_labels.append(labels.cpu())

        #TODO: remove this break
        if len(all_embs) >= 100:  # Just process 100 batches for quick testing
            break

    all_embs = torch.cat(all_embs, dim=0)    # [N, D]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    return all_embs, all_labels


# =========================
# Compute ROC from embeddings
# =========================
def compute_roc(query_embs, query_labels, gallery_embs, gallery_labels,
                balanced=False, chunk_size=512, max_roc_points=10000, seed=42):
    """
    Compute ROC curve and AUC.

    balanced=False: all NxM pairs (imbalanced).
    balanced=True:  for each query, sample 1 positive and 1 negative gallery match (balanced).
    """
    q_embs = query_embs.numpy() if isinstance(query_embs, torch.Tensor) else query_embs
    q_labels = query_labels.numpy() if isinstance(query_labels, torch.Tensor) else query_labels
    g_embs = gallery_embs.numpy() if isinstance(gallery_embs, torch.Tensor) else gallery_embs
    g_labels = gallery_labels.numpy() if isinstance(gallery_labels, torch.Tensor) else gallery_labels

    print(f"Computing roc with balanced={balanced} ...")
    if balanced:
        rng = np.random.RandomState(seed)
        # build label -> gallery indices map
        label_to_idx = {}
        for i, lbl in enumerate(g_labels):
            label_to_idx.setdefault(int(lbl), []).append(i)
        all_gallery_labels = set(label_to_idx.keys())

        scores = []
        labels = []
        for i in range(len(q_embs)):
            qlbl = int(q_labels[i])
            pos_indices = label_to_idx.get(qlbl, [])
            neg_labels = list(all_gallery_labels - {qlbl})
            if len(pos_indices) == 0 or len(neg_labels) == 0:
                raise ValueError(f"Query label {qlbl} has no positive or negative matches in gallery.")
            # sample 1 positive
            pos_idx = rng.choice(pos_indices)
            # sample 1 negative
            neg_lbl = rng.choice(neg_labels)
            neg_idx = rng.choice(label_to_idx[neg_lbl])
            # cosine similarity (embeddings already L2-normalized)
            scores.append(q_embs[i] @ g_embs[pos_idx])
            labels.append(1)
            scores.append(q_embs[i] @ g_embs[neg_idx])
            labels.append(0)

        scores = np.array(scores, dtype=np.float32)
        labels = np.array(labels, dtype=np.int8)
    else:
        all_scores = []
        all_labels = []
        for i in range(0, len(q_embs), chunk_size):
            chunk_embs = q_embs[i:i+chunk_size]
            chunk_labels = q_labels[i:i+chunk_size]
            chunk_scores = (chunk_embs @ g_embs.T).reshape(-1)
            chunk_match = (chunk_labels[:, None] == g_labels[None, :]).reshape(-1).astype(np.int8)
            all_scores.append(chunk_scores)
            all_labels.append(chunk_match)
        scores = np.concatenate(all_scores)
        del all_scores
        labels = np.concatenate(all_labels)
        del all_labels

    gc.collect()
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    del scores, labels
    gc.collect()

    if len(fpr) > max_roc_points:
        idx = np.linspace(0, len(fpr) - 1, max_roc_points, dtype=int)
        fpr = fpr[idx]
        tpr = tpr[idx]

    return fpr, tpr, roc_auc


# =========================
# Main
# =========================
def main():
    plt.figure(figsize=(8, 8))

    # Load existing progress
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            progress = json.load(f)
        print(f"Loaded progress from {LOG_FILE}: ratios {list(progress.keys())} already computed.")
    else:
        progress = {}

    # Extract fullface embeddings once using the first ratio's config
    print("Extracting fullface embeddings (once)...")
    _, cfg0 = load_partial_model(RATIOS[0])
    fullface_model = load_fullface_model(cfg0)
    full_embs, full_labels = extract_embeddings(fullface_model, FULL_DIR)
    del fullface_model
    torch.cuda.empty_cache()
    gc.collect()

    for ratio in RATIOS:
        ratio_key = str(ratio)

        # Skip if already computed
        if ratio_key in progress:
            print(f"\nRatio {ratio}% already computed (AUC={progress[ratio_key]['auc']:.6f}), skipping.")
            continue

        print(f"\n{'='*40}")
        print(f"Processing ratio {ratio}%")
        print(f"{'='*40}")

        partial_dir = f"/datasets/glint360k/ROIs/ratio_{ratio}/test"

        partial_face_model, _ = load_partial_model(ratio)
        partial_embs, partial_labels = extract_embeddings(partial_face_model, partial_dir)

        # free GPU memory before computing scores
        del partial_face_model
        torch.cuda.empty_cache()

        fpr, tpr, roc_auc = compute_roc(
            partial_embs, partial_labels, full_embs, full_labels
        )
        del partial_embs, partial_labels
        gc.collect()

        print(f"Ratio {ratio}% — AUC = {roc_auc:.6f}")

        # Save progress for this ratio
        progress[ratio_key] = {
            "auc": float(roc_auc),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
        with open(LOG_FILE, "w") as f:
            json.dump(progress, f)
        print(f"Progress saved to {LOG_FILE}")

        del fpr, tpr
        gc.collect()

    # Plot all ratios (both cached and newly computed)
    for ratio in RATIOS:
        ratio_key = str(ratio)
        if ratio_key in progress:
            entry = progress[ratio_key]
            fpr = np.array(entry["fpr"])
            tpr = np.array(entry["tpr"])
            roc_auc = entry["auc"]
            plt.plot(fpr, tpr, label=f"Ratio {ratio}% (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: Partial Face vs Full Face — Multiple Ratios")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_ROC_IMG, dpi=200)
    plt.close()

    print(f"\nROC curve saved to: {OUT_ROC_IMG}")


if __name__ == "__main__":
    main()