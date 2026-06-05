"""
Benchmark CLIP-trained partial encoders.
For each ROI ratio, load the CLIP-trained ViT-S partial encoder and the
full-face teacher (r50). Evaluate with ClipVerification (verification acc + rank-1).
"""
import gc
import json
import os
import sys

import torch
import matplotlib.pyplot as plt

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
RATIOS = [100, 60, 40, 30, 25, 20, 15]
OUT_IMG = "work_dirs/clip_encoder_accuracy.png"
LOG_FILE = "work_dirs/clip_encoder_accuracy.json"
WORK_DIR = "work_dirs"
MAX_EMBEDDINGS = 1000
BATCH_SIZE = 64

# =========================
# Main
# =========================
def main():
    # if os.path.exists(LOG_FILE):
    #     with open(LOG_FILE, "r") as f:
    #         progress = json.load(f)
    #     print(f"Loaded progress from {LOG_FILE}: ratios {list(progress.keys())} already computed.")
    # else:
    progress = {}

    # Load full-face teacher model (shared across all ratios)
    _, cfg0 = load_partial_model(RATIOS[0])
    print(f"Loading full-face teacher model from {cfg0.teacher_model_path} ...")
    teacher_model = load_fullface_model(cfg0)

    for ratio in RATIOS:
        ratio_key = str(ratio)
        # if ratio_key in progress:
        #     print(f"Skipping ratio {ratio}% (already computed)")
        #     continue

        print(f"\n{'='*40}")
        print(f"Processing ratio {ratio}%")
        print(f"{'='*40}")

        # Load the CLIP-trained partial encoder for this ratio
        partial_model, cfg = load_partial_model(ratio)

        partial_dir = f"/datasets/glint360k/ROIs/ratio_{ratio}/test"

        verifier = ClipVerification(
            val_targets=[partial_dir, FULL_DIR],
            train_targets=[partial_dir, FULL_DIR],
            batch_size=BATCH_SIZE,
            work_dir=WORK_DIR,
        )

        best_acc, rank1 = verifier(
            backbone_partial=partial_model,
            backbone_full=teacher_model,
            global_step=0,
            epoch=ratio,
            max_embeddings=MAX_EMBEDDINGS,
        )

        print(f"Ratio {ratio}% — BestAcc={best_acc:.4f}  Rank1={rank1:.4f}")

        progress[ratio_key] = {
            "best_acc": float(best_acc),
            "rank1": float(rank1),
        }
        with open(LOG_FILE, "w") as f:
            json.dump(progress, f, indent=2)
        print(f"Progress saved to {LOG_FILE}")

        del verifier, partial_model
        gc.collect()
        torch.cuda.empty_cache()

    del teacher_model
    torch.cuda.empty_cache()

    # ---- Plot ----
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
    plt.title("CLIP-Trained Partial Encoder — Accuracy vs ROI Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=200)
    plt.close()
    print(f"\nPlot saved to: {OUT_IMG}")


if __name__ == "__main__":
    main()
