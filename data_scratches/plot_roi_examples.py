"""Visual sanity-check: a few example identities cropped at each ROI ratio,
side by side, to illustrate what the sweep's degradation actually looks like.
Purely for human inspection -- not a training artifact.
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import matplotlib.pyplot as plt

from dataset import crop_roi
from training_arch_comparison import VAL_DIR, VAL_METADATA_PATH

ROI_RATIOS = [1.0, 0.6, 0.4, 0.3, 0.2, 0.15]
N_IDENTITIES = 4
SEED = 11
OUT_PATH = "work_dirs/visual_inspectioin/roi_examples.png"


def pick_examples(metadata, n, seed):
    by_id = {}
    for k, v in metadata.items():
        if v.get("detected", False):
            by_id.setdefault(k.split("/", 1)[0], []).append(k)
    rng = random.Random(seed)
    ids = list(by_id.keys())
    rng.shuffle(ids)
    chosen = ids[:n]
    return [rng.choice(by_id[id_]) for id_ in chosen]


def main():
    metadata = json.load(open(VAL_METADATA_PATH))
    keys = pick_examples(metadata, N_IDENTITIES, SEED)

    fig, axes = plt.subplots(N_IDENTITIES, len(ROI_RATIOS), figsize=(2.2 * len(ROI_RATIOS), 2.4 * N_IDENTITIES))
    for row, key in enumerate(keys):
        id_name, filename = key.split("/", 1)
        bgr = cv2.imread(os.path.join(VAL_DIR, id_name, filename))
        center_y = metadata[key]["center_y"]
        for col, ratio in enumerate(ROI_RATIOS):
            crop = crop_roi(bgr, ratio, center_y, 1.0)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ax = axes[row, col]
            ax.imshow(rgb)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                label = "100% (full)" if ratio == 1.0 else f"{int(ratio * 100)}%"
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"id {row}", fontsize=10)

    fig.suptitle("ROI crop examples across ratios (same identities, decreasing visible face height)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
