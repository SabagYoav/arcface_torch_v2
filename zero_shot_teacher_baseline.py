"""Zero-shot teacher baseline: no training. Measures how well the ALREADY-
TRAINED R50 teacher (16backbone.pth) verifies identity when given a
partial-face probe (ROI-cropped, same crop_roi() as training) against a
full-face gallery, using the SAME teacher model as both `backbone_partial`
and `backbone_full` embedder in ClipVerification. This is the "before
distillation" reference curve -- pairs with the distilled swin_base student's
degradation curve (training_fast_roi_sweep.py) to show the method's benefit
as the gap between the two curves.
"""
import json
import os

import torch
import matplotlib.pyplot as plt

from backbones import get_model
from utils.clip_verifications_utils import ClipVerification

TEACHER_CKPT = "16backbone.pth"
ROI_RATIOS = [0.15, 0.2, 0.3, 0.4, 0.6, 1.0]
EXP_ROOT = "work_dirs/exp_zero_shot_teacher_baseline"
OUT_JSON = os.path.join(EXP_ROOT, "results.json")
OUT_PLOT = os.path.join(EXP_ROOT, "zero_shot_degradation.png")


def tag_for(ratio: float) -> str:
    return f"ratio_{int(round(ratio * 100))}"


def main():
    os.makedirs(EXP_ROOT, exist_ok=True)

    results = {}
    if os.path.exists(OUT_JSON):
        results = json.load(open(OUT_JSON))

    missing = [r for r in ROI_RATIOS if tag_for(r) not in results]
    if missing:
        teacher = get_model("r50", dropout=0.0, fp16=True, num_features=512).cuda()
        teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location="cuda"))
        teacher.eval()

        for ratio in missing:
            tag = tag_for(ratio)
            print(f"Computing zero-shot teacher baseline at ratio={ratio}...")
            verifier = ClipVerification(ratio=ratio, work_dir=EXP_ROOT)
            best_acc, acc_mean = verifier(teacher, teacher, global_step=0, epoch=0)
            results[tag] = {"best_acc": best_acc, "acc_mean": acc_mean}
            json.dump(results, open(OUT_JSON, "w"))
            print(f"  {tag}: best_acc={best_acc:.4f} acc_mean={acc_mean:.4f}")
    else:
        print("All ratios already computed, skipping teacher load.")

    points = sorted((int(k.rsplit("_", 1)[1]), v["best_acc"]) for k, v in results.items())
    plt.figure(figsize=(8, 5))
    plt.plot([p[0] for p in points], [p[1] for p in points],
              marker="o", color="tab:gray", label="Zero-shot teacher (no distillation)")
    plt.xlabel("ROI ratio (% of face height visible)")
    plt.ylabel("Best verification accuracy (LFW cross_partial_vs_full)")
    plt.title("Zero-shot teacher baseline degradation")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved -> {OUT_PLOT}")


if __name__ == "__main__":
    main()
