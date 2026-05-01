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
OUT_ROC_IMG = "work_dirs/roc_partial_vs_full_multi.png"
LOG_FILE = "work_dirs/roc_multi_progress.json"

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

        # convert to numpy once
        partial_embs_np = partial_embs.numpy()
        partial_labels_np = partial_labels.numpy()
        full_embs_np = full_embs.numpy()
        full_labels_np = full_labels.numpy()
        del partial_embs, partial_labels

        # compute scores and labels in chunks to avoid N×N matrix in memory
        CHUNK = 512
        all_scores = []
        all_labels_list = []
        for i in range(0, len(partial_embs_np), CHUNK):
            chunk_embs = partial_embs_np[i:i+CHUNK]
            chunk_labels = partial_labels_np[i:i+CHUNK]
            chunk_scores = (chunk_embs @ full_embs_np.T).reshape(-1)
            chunk_match = (chunk_labels[:, None] == full_labels_np[None, :]).reshape(-1).astype(np.int8)
            all_scores.append(chunk_scores)
            all_labels_list.append(chunk_match)

        del partial_embs_np, partial_labels_np, full_embs_np, full_labels_np
        scores = np.concatenate(all_scores)
        del all_scores
        labels = np.concatenate(all_labels_list)
        del all_labels_list
        gc.collect()

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # free large arrays
        del scores, labels
        gc.collect()

        # Subsample ROC curve to keep file size manageable
        MAX_ROC_POINTS = 10000
        if len(fpr) > MAX_ROC_POINTS:
            idx = np.linspace(0, len(fpr) - 1, MAX_ROC_POINTS, dtype=int)
            fpr = fpr[idx]
            tpr = tpr[idx]

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