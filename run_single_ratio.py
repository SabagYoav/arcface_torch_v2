"""Run a single ROI ratio of the main sweep on demand (e.g. in parallel on a
different GPU from the sequential training_multi_loops.py sweep). Writes into
the SAME EXP_NAME/output tree, so the main sweep will see this ratio's
result.json and skip it once it reaches this ratio in its own loop.

Usage: CUDA_VISIBLE_DEVICES=1 python3 run_single_ratio.py <ratio>
"""
import sys

from dataset import load_eye_center_metadata
from training_multi_loops import (
    EYE_CENTER_METADATA_PATH, load_ratio_result, save_ratio_result,
    read_and_update_variant_config, run_training,
)


def main():
    ratio = float(sys.argv[1])
    tag = f"ratio_{int(ratio * 100)}"

    cached = load_ratio_result(tag)
    if cached is not None:
        print(f"{tag} already complete: {cached}")
        return

    print("Loading eye-center metadata (full glint360k)...")
    metadata = load_eye_center_metadata(EYE_CENTER_METADATA_PATH)
    print(f"Loaded {len(metadata)} metadata entries.")

    args = read_and_update_variant_config(tag, ratio=ratio, metadata=metadata, exp_name_prefix="clip")
    print(f"Training {tag}...")
    res = run_training(args)
    save_ratio_result(tag, res)
    print(f"Completed {tag}: {res}")


if __name__ == "__main__":
    main()
