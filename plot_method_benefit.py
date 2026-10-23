"""Combined figure: distilled swin_base student vs zero-shot teacher baseline,
across the same 6 ROI ratios -- demonstrates the method's benefit as the gap
between the two degradation curves, without needing 500+ hours of full-sweep
training.
"""
import json
import os
import matplotlib.pyplot as plt

from training_arch_comparison import EXP_ROOT as ARCH_EXP_ROOT

STUDENT_RESULTS = os.path.join(ARCH_EXP_ROOT, "swin_base_roi_sweep_results.json")
TEACHER_RESULTS = "work_dirs/exp_zero_shot_teacher_baseline/results.json"
OUT_PLOT = "work_dirs/exp_zero_shot_teacher_baseline/method_benefit_comparison.png"


def main():
    student = json.load(open(STUDENT_RESULTS))
    teacher = json.load(open(TEACHER_RESULTS))

    student_points = sorted((int(k.rsplit("_", 1)[1]), v["best_acc"]) for k, v in student.items())
    teacher_points = sorted((int(k.rsplit("_", 1)[1]), v["best_acc"]) for k, v in teacher.items())

    plt.figure(figsize=(8, 5))
    plt.plot([p[0] for p in student_points], [p[1] for p in student_points],
              marker="o", color="tab:brown", label="PA-Net")
    plt.plot([p[0] for p in teacher_points], [p[1] for p in teacher_points],
              marker="o", color="tab:gray", linestyle="--", label="Zero-shot teacher")
    plt.xlabel("ROI ratio (%)")
    plt.ylabel("Verification Accuracy")
    plt.title("Method benefit: distilled student vs. zero-shot teacher baseline")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    plt.savefig(OUT_PLOT, dpi=150)
    plt.close()
    print(f"Saved -> {OUT_PLOT}")


if __name__ == "__main__":
    main()
