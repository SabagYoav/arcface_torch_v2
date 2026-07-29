import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from train_v4_clip import train
from utils.utils_config import get_config
from training_arch_comparison import (
    TRAIN_DIR, VAL_DIR, TRAIN_METADATA_PATH, VAL_METADATA_PATH,
    build_scoped_metadata, write_variants_config,
)

# Temperature analysis: fix the winning architecture (swin_base) and a single
# ROI ratio, vary only the CLIP loss temperature. Reuses the exact same
# 36k-train/9k-val (10-instances/id capped) subset + metadata as
# training_arch_comparison.py -- no new data prep needed. The 0.07 baseline
# already exists at work_dirs/exp_arch_comparison_10perid/swin_base_ratio_15
# (best_acc=0.982) and is pulled into the comparison plot, not retrained.
TEACHER_MODEL_PATH = Path("16backbone.pth")
BASE_CONFIG = "configs/experiment_arch_comparison_clip.py"
EXP_NAME = os.environ.get("EXP_NAME", "exp_temperature_analysis")
EXP_ROOT = f"work_dirs/{EXP_NAME}"

NETWORK = "swin_base"
BATCH_SIZE = 448   # matches swin_base's setting in the arch comparison, for a fair comparison
ROI_RATIO = 0.15

TEMPERATURES = [0.03, 0.05, 0.10, 0.15]
BASELINE_TEMPERATURE = 0.07
BASELINE_RESULT_PATH = Path("work_dirs/exp_arch_comparison_10perid/swin_base_ratio_15/result.json")


def tag_for(temperature: float) -> str:
    return f"temp_{int(round(temperature * 100)):03d}"


def output_dir(temperature: float) -> Path:
    return Path(EXP_ROOT) / tag_for(temperature)


def load_result(temperature: float):
    p = output_dir(temperature) / "result.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def save_result(temperature: float, res: dict):
    out = output_dir(temperature)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "result.json", "w") as f:
        json.dump(res, f)


def build_config(temperature: float, num_classes: int, num_image: int):
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", default=BASE_CONFIG, type=str)
    args = parser.parse_args([])

    cfg = get_config(args.config)
    cfg.network = NETWORK
    cfg.batch_size = BATCH_SIZE
    cfg.use_onthefly_dataset = True
    cfg.root_dir = str(TRAIN_DIR)
    cfg.eye_center_metadata_path = str(TRAIN_METADATA_PATH)
    cfg.roi_ratio = ROI_RATIO
    cfg.teacher_model_path = str(TEACHER_MODEL_PATH)
    cfg.glint_val_root = str(VAL_DIR)
    cfg.glint_val_metadata_path = str(VAL_METADATA_PATH)
    cfg.temperature = temperature
    cfg.output = str(output_dir(temperature))
    cfg.num_classes = num_classes
    cfg.num_image = num_image

    cfg_path = Path(f"configs/variants_temperature_{tag_for(temperature)}.py")
    write_variants_config(cfg_path, cfg)

    args.config = str(cfg_path)
    return args


def plot_comparison(results: dict, filename: str = None):
    filename = filename or os.path.join(EXP_ROOT, "temperature_comparison.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    points = sorted(results.items())
    xs = [t for t, _ in points]
    ys = [r.get("best_acc", float("nan")) for _, r in points]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", color="tab:brown")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.xlabel("CLIP loss temperature")
    plt.ylabel("Best verification accuracy (LFW cross_partial_vs_full)")
    plt.title(f"Temperature analysis — {NETWORK}, ROI ratio {ROI_RATIO}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved comparison plot -> {filename}")


def main():
    build_scoped_metadata()

    from dataset import load_eye_center_metadata
    train_metadata = load_eye_center_metadata(TRAIN_METADATA_PATH)
    detected = [k for k, v in train_metadata.items() if v.get("detected", False)]
    num_classes = len({k.split("/", 1)[0] for k in detected})
    num_image = len(detected)
    print(f"Training subset: {num_classes} identities, {num_image} images")
    del train_metadata, detected

    results = {}
    if BASELINE_RESULT_PATH.exists():
        results[BASELINE_TEMPERATURE] = json.load(open(BASELINE_RESULT_PATH))
        print(f"Baseline temperature={BASELINE_TEMPERATURE} (reused, not retrained): {results[BASELINE_TEMPERATURE]}")

    for temperature in TEMPERATURES:
        cached = load_result(temperature)
        if cached is not None:
            print(f"Skipping temperature={temperature} (already complete): {cached}")
            results[temperature] = cached
            continue

        args = build_config(temperature, num_classes, num_image)
        print(f"Training temperature={temperature}...")
        res = train(args)
        save_result(temperature, res)
        results[temperature] = res
        print(f"Completed temperature={temperature}: {res}")

    plot_comparison(results)


if __name__ == "__main__":
    main()
