"""Overlay best verification accuracy vs ROI ratio for multiple CLIP experiments.

Reads work_dirs/<exp>/clip_ratio_<XX>/result.json for each experiment and plots
one line per experiment on shared axes, so different students (e.g. R50 vs ViT)
trained against the same R50 teacher can be compared directly.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROI_RATIOS = [1.0, 0.6, 0.4, 0.3, 0.2, 0.15]

# (work_dir name, legend label)
EXPERIMENTS = [
    ("exp_r50_vs_r50", "R50 student"),
    ("exp_r50_vs_vit", "ViT-S student"),
]
OUT_PATH = "work_dirs/r50_teacher_student_comparison.png"


def load_curve(exp_dir):
    xs, ys = [], []
    for ratio in ROI_RATIOS:
        tag = f"ratio_{int(round(ratio * 100))}"
        p = os.path.join("work_dirs", exp_dir, f"clip_{tag}", "result.json")
        if not os.path.exists(p):
            continue
        try:
            acc = json.load(open(p)).get("best_acc")
        except Exception:
            continue
        if acc is None:
            continue
        xs.append(int(round(ratio * 100)))
        ys.append(acc)
    # sort by ROI ratio ascending
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


def main():
    plt.figure(figsize=(8, 5))
    plotted = 0
    for exp_dir, label in EXPERIMENTS:
        xs, ys = load_curve(exp_dir)
        if not xs:
            print(f"⚠️  no results found for {exp_dir}, skipping")
            continue
        plt.plot(xs, ys, marker="o", linewidth=2, label=label)
        for x, y in zip(xs, ys):
            plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
        plotted += 1
        print(f"{label}: " + ", ".join(f"{x}%={y:.4f}" for x, y in zip(xs, ys)))

    if plotted == 0:
        print("No experiment results to plot.")
        return

    plt.xlabel("ROI ratio (% of face height visible)")
    plt.ylabel("Best verification accuracy (val)")
    plt.title("CLIP distillation vs ROI — R50 teacher, R50 vs ViT-S student")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_xaxis()   # full face (100) left, eyes-only (15) right
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    plt.close()
    print(f"✅ Saved comparison plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
