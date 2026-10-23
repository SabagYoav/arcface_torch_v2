"""Plot training-loss curves from the architecture comparison's per-run
training.log files -- one subplot per ROI ratio, one line per architecture,
averaged to one point per epoch.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt

EXP_ROOT = Path("work_dirs/exp_arch_comparison_10perid")
ARCHITECTURES = ["r50", "r100", "vit_b16", "mobilevit_s", "swin_base"]
ROI_RATIOS = [0.15, 0.2, 0.4, 0.6]

# Kept in sync with training_arch_comparison.py's ARCH_COLORS so the same
# architecture is the same color in both the accuracy and loss figures.
ARCH_COLORS = {
    "r50": "tab:blue",
    "r100": "tab:orange",
    "vit_b16": "tab:green",
    "swin_tiny": "tab:red",
    "mobilevit_s": "tab:purple",
    "swin_base": "tab:brown",
}

LINE_RE = re.compile(r"Loss ([\d.]+)\s+LearningRate .*?Epoch: (\d+)")


def parse_loss_by_epoch(log_path):
    """Returns (epochs, mean_loss_per_epoch), one point per epoch."""
    if not log_path.exists():
        return [], []
    per_epoch = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = LINE_RE.search(line)
        if m:
            loss, epoch = float(m.group(1)), int(m.group(2))
            per_epoch.setdefault(epoch, []).append(loss)
    epochs = sorted(per_epoch)
    means = [sum(per_epoch[e]) / len(per_epoch[e]) for e in epochs]
    return epochs, means


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
    axes = axes.flatten()

    for ax, ratio in zip(axes, ROI_RATIOS):
        tag = f"ratio_{int(round(ratio * 100))}"
        any_data = False
        for arch in ARCHITECTURES:
            log_path = EXP_ROOT / f"{arch}_{tag}" / "training.log"
            epochs, losses = parse_loss_by_epoch(log_path)
            if not epochs:
                continue
            any_data = True
            ax.plot(epochs, losses, label=arch, marker="o", linewidth=1.5, color=ARCH_COLORS.get(arch))
        ax.set_title(f"ROI ratio {ratio}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CLIP loss (mean per epoch)")
        ax.grid(True, alpha=0.3)
        if any_data:
            ax.legend(fontsize=8)

    fig.suptitle("Architecture comparison — training loss curves (10 instances/id subset)")
    fig.tight_layout()
    out_path = EXP_ROOT / "arch_comparison_losses.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
