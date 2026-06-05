"""
Baseline experiment: Full-face encoder for both gallery and query.
Gallery = full-face images, Query = partial-face images at varying occlusion ratios.
Both encoded by the same full-face trained model (r50).

Uses ClipVerification.__call__ from utils/clip_verifications_utils.py
which bundles paired embedding extraction, verification accuracy, and rank-1.
"""
import gc
import json
import os
import sys

import torch
import matplotlib.pyplot as plt

# Ensure eval/ and project root are importable
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, ".."))

from roc_curve_multi import (
    load_fullface_model,
    load_partial_model,
    FULL_DIR,
)
from utils.clip_verifications_utils import ClipVerification


# =========================
# Config
# =========================
RATIOS = [100, 60, 40, 30, 20, 15]
OUT_IMG = "work_dirs/baseline_fullface_encoder_accuracy.png"
LOG_FILE = "work_dirs/baseline_fullface_encoder_accuracy.json"
WORK_DIR = "work_dirs"
MAX_EMBEDDINGS = 1000
BATCH_SIZE = 64


# =========================
# Main
# =========================
def main():
    ## Load existing progress ##
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            progress = json.load(f)
        print(f"Loaded progress from {LOG_FILE}: ratios {list(progress.keys())} already computed.")
    else:
        progress = {}

    # Load the full-face (teacher) model via roc_curve_multi helpers
    _, cfg = load_partial_model(RATIOS[0])
    print(f"Loading full-face model from {cfg.teacher_model_path} ...")
    model = load_fullface_model(cfg)

    for ratio in RATIOS:
        ratio_key = str(ratio)

        print(f"\n{'='*40}")
        print(f"Processing ratio {ratio}%")
        print(f"{'='*40}")

        partial_dir = f"/datasets/glint360k/ROIs/ratio_{ratio}/test"

        # ClipVerification pairs partial ↔ full via its clip dataloader
        verifier = ClipVerification(
            val_targets=[partial_dir, FULL_DIR],
            train_targets=[partial_dir, FULL_DIR],
            batch_size=BATCH_SIZE,
            work_dir=WORK_DIR,
        )

        # __call__ extracts paired embeddings and computes verification + rank-1
        # Pass the same full-face model as both backbone_partial and backbone_full
        best_acc, rank1 = verifier(
            backbone_partial=model,
            backbone_full=model,
            global_step=0,
            epoch=ratio,
            max_embeddings=MAX_EMBEDDINGS,
        )

        print(f"Ratio {ratio}% — BestAcc={best_acc:.4f}  Rank1={rank1:.4f}")

        # Save progress
        progress[ratio_key] = {
            "best_acc": float(best_acc),
            "rank1": float(rank1),
        }
        with open(LOG_FILE, "w") as f:
            json.dump(progress, f, indent=2)
        print(f"Progress saved to {LOG_FILE}")

        del verifier
        gc.collect()
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    # ---- Plot: Accuracy & Rank-1 vs ROI ratio ----
    sorted_ratios = sorted([r for r in RATIOS if str(r) in progress])
    best_accs = [progress[str(r)]["best_acc"] for r in sorted_ratios]
    rank1s = [progress[str(r)]["rank1"] for r in sorted_ratios]

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_ratios, best_accs, "o-", linewidth=2, markersize=8, label="Verification Acc")
    plt.plot(sorted_ratios, rank1s, "s--", linewidth=2, markersize=8, label="Rank-1 Acc")
    for r, a, r1 in zip(sorted_ratios, best_accs, rank1s):
        plt.annotate(f"{a:.4f}", (r, a), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
        plt.annotate(f"{r1:.4f}", (r, r1), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    plt.xlabel("ROI Ratio (%)")
    plt.ylabel("Accuracy")
    plt.title("Baseline: Full-Face Encoder — Accuracy vs ROI Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=200)
    plt.close()
    print(f"\nAccuracy plot saved to: {OUT_IMG}")


if __name__ == "__main__":
    main()
