"""Fast 6-point ROI degradation sweep for the winning architecture (swin_base),
on the small subset (10 instances/id) used by training_arch_comparison.py --
reuses that script's already-completed ratios [0.15, 0.2, 0.4, 0.6] and only
trains the 2 missing ones [0.3, 1.0], so the full 6-ratio curve costs ~2 runs
instead of 6 full-dataset runs. Paired with zero_shot_teacher_baseline.py (no
training) to show the method's benefit as the gap between the distilled
student's curve and the untrained teacher's curve on the same partial-face
crops.
"""
import json
import os

from train_v4_clip import train
from dataset import load_eye_center_metadata
from training_arch_comparison import (
    build_scoped_metadata, build_config, load_result, save_result,
    tag_for, EXP_ROOT, COMMON_BATCH_SIZE, TRAIN_METADATA_PATH,
)

NETWORK = "swin_base"
ROI_RATIOS = [0.15, 0.2, 0.3, 0.4, 0.6, 1.0]


def main():
    build_scoped_metadata()

    train_metadata = load_eye_center_metadata(TRAIN_METADATA_PATH)
    detected = [k for k, v in train_metadata.items() if v.get("detected", False)]
    num_classes = len({k.split("/", 1)[0] for k in detected})
    num_image = len(detected)
    print(f"Training subset: {num_classes} identities, {num_image} images")
    del train_metadata, detected

    results = {}
    for ratio in ROI_RATIOS:
        tag = tag_for(NETWORK, ratio)
        cached = load_result(NETWORK, ratio)
        if cached is not None:
            print(f"Skipping {tag} (already complete): {cached}")
            results[tag] = cached
            continue

        args = build_config(NETWORK, COMMON_BATCH_SIZE, ratio, num_classes, num_image)
        print(f"Training {tag}...")
        res = train(args)
        save_result(NETWORK, ratio, res)
        results[tag] = res
        print(f"Completed {tag}: {res}")

    out_path = os.path.join(EXP_ROOT, "swin_base_roi_sweep_results.json")
    json.dump(results, open(out_path, "w"))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
