import os
import json
import random
import logging
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from train_v4_clip import train
from utils.utils_config import get_config
from dataset import OnTheFlyClipDataset, load_eye_center_metadata, crop_roi

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
# Full glint360k dataset (360,232 identities, ~21M images) + buffalo_l eye-center
# metadata (see data_scratches/build_eye_center_metadata.py). ROI crops are made
# on-the-fly from this metadata (OnTheFlyClipDataset) instead of duplicating the
# dataset on disk once per ratio — untenable at this scale (166GB x 6 ratios).
GLINT_ROOT = Path("/DATA/glint360k_download/glint360k_images")
EYE_CENTER_METADATA_PATH = Path("/DATA/glint360k_download/eye_center_metadata/center_eyes_metadata.json")
TEACHER_MODEL_PATH = Path("16backbone.pth")  # repo root

# Experiment config + output namespace. Override per experiment via env vars, e.g.
#   EXP_NAME=exp_r50_vs_r50_clip BASE_CONFIG=configs/experiment_r50_vs_r50_clip.py python training_multi_loops.py
# so each experiment writes to its own work_dirs/<EXP_NAME>/ folder (and its
# result.json markers only block reruns of that same experiment).
BASE_CONFIG = 'configs/experiment_r50_vs_vit_clip.py'
# Distinct from the completed subset sweep's "exp_clip_r50_vs_vit" — must not
# collide, or the resume-skip logic below would see those result.json markers
# and think this (full-dataset) sweep is already done.
EXP_NAME = os.environ.get("EXP_NAME", "exp_clip_r50_vs_vit_full")

ROI_RATIOS = [0.15, 0.2, 0.3, 0.4, 0.6, 1.0]
ROI_WIDTH_RATIO = 1.0   # relative to face width

ACCURACY_PLOT_PATH = os.path.join("work_dirs", EXP_NAME, "fullset_accuracy_comparison.png")
VISUAL_INSPECTION_DIR = Path("work_dirs/visual_inspectioin")

TRAINING_FLAG = True


def setup_multi_loops_logger(log_path="training_multi_loops_log.txt"):
    logger = logging.getLogger("multi_loops_logger")
    logger.setLevel(logging.INFO)
    # Prevent duplicate logs if this function is called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def plot_metric_vs_variant(results: dict, metric: str, filename: str):
    items = sorted(results.items(), key=lambda x: x[0])

    variants = [tag for tag, _ in items]
    metric_values = [data[metric] for _, data in items]

    print(f"Plotting {metric} for variants: {variants} with values: {metric_values}")

    plt.figure(figsize=(8, 5))
    plt.plot(variants, metric_values, marker="o")
    plt.xlabel("ROI variant")
    plt.ylabel(metric)
    plt.title(f"{metric} vs ROI variant")
    plt.grid(True)
    plt.gca().invert_xaxis()   # flip X axis
    plt.savefig(filename)
    plt.close()


def plot_results(results: dict, out_dir="variants_plots"):
    os.makedirs(out_dir, exist_ok=True)

    metrics = next(iter(results.values())).keys()

    for metric in metrics:
        plot_metric_vs_variant(
            results,
            metric=metric,
            filename=os.path.join(out_dir, f"roi_sweep_{metric}.png"),
        )


def plot_accuracy_comparison(results: dict, filename: str = ACCURACY_PLOT_PATH):
    """Combined plot: best verification accuracy vs ROI ratio (R50 student vs R50 teacher)."""
    if not results:
        print("No results to plot.")
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    items = sorted(results.items(), key=lambda x: x[0])              # by ratio
    ratios = [r for r, _ in items]
    accs = [d.get("best_acc", float("nan")) for _, d in items]
    xs = [int(round(r * 100)) for r in ratios]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, accs, marker="o", linewidth=2)
    for x, a in zip(xs, accs):
        plt.annotate(f"{a:.3f}", (x, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.xlabel("ROI ratio (% of face height visible)")
    plt.ylabel("Best verification accuracy (LFW cross_partial_vs_full)")
    plt.title("R50 student vs R50 teacher — CLIP distillation accuracy vs ROI")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()   # full face (100) on the left, eyes-only (15) on the right
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved accuracy comparison plot to {filename}")


def save_visual_inspection_grid(metadata, ratio, tag, n_samples=8, seed=0):
    """Side-by-side full-face / on-the-fly partial-face crop grid, for a quick
    visual sanity check before committing to training on a given ratio."""
    import cv2

    detected_items = [k for k, v in metadata.items() if v.get("detected", False)]
    rng = random.Random(seed)
    sample_keys = rng.sample(detected_items, min(n_samples, len(detected_items)))

    rows = []
    for key in sample_keys:
        id_name, filename = key.split("/", 1)
        bgr = cv2.imread(str(GLINT_ROOT / id_name / filename))
        if bgr is None:
            continue
        partial = crop_roi(bgr, ratio, metadata[key]["center_y"], ROI_WIDTH_RATIO)
        rows.append(np.hstack([bgr, partial]))

    if not rows:
        print(f"⚠️  No samples available for visual inspection grid ({tag}).")
        return

    grid = np.vstack(rows)
    VISUAL_INSPECTION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VISUAL_INSPECTION_DIR / f"sample_grid_{tag}.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"🖼️  Saved visual inspection grid ({len(rows)} samples, full|partial) -> {out_path}")


def read_and_update_variant_config(tag: str, ratio: float, metadata, exp_name_prefix=None):
    import torch
    exp_name = f"{exp_name_prefix}_{tag}"
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch"
    )
    parser.add_argument("--config", default=BASE_CONFIG, type=str)
    args = parser.parse_args([])

    cfg = get_config(args.config)
    cfg.use_onthefly_dataset = True
    cfg.root_dir = str(GLINT_ROOT)
    cfg.eye_center_metadata_path = str(EYE_CENTER_METADATA_PATH)
    cfg.roi_ratio = ratio
    cfg.teacher_model_path = str(TEACHER_MODEL_PATH)
    cfg.output = str(ratio_output_dir(tag))
    cfg.batch_grid_tag = exp_name

    detected_items = [k for k, v in metadata.items() if v.get("detected", False)]
    cfg.num_classes = len({k.split("/", 1)[0] for k in detected_items})
    cfg.num_image = len(detected_items)

    cfg_path = Path(f"configs/variants_config_{tag}.py")
    write_variants_config(dst_root=cfg_path, cfg=cfg)

    args.config = str(cfg_path)
    return args


def run_training(args):
    results_dict = train(args)
    return results_dict


def write_variants_config(dst_root: Path, cfg: dict):
    with open(dst_root, "w") as f:
        f.write("# Auto-generated config file\n")
        f.write("from easydict import EasyDict as edict\n\n")
        f.write("config = edict()\n")

        for key, value in cfg.items():
            f.write(f"config.{key} = {repr(value)}\n")


# -------------------------------------------------
# Resume helpers (per-ratio completion markers)
# -------------------------------------------------
# Dedicated output namespace so this experiment can never collide with stale
# checkpoints from other experiments (which would corrupt train_v4's resume).
# Derived from EXP_NAME (set at top) so the plot path and results dir always agree.
EXP_ROOT = f"work_dirs/{EXP_NAME}"


def ratio_output_dir(tag: str) -> Path:
    return Path(EXP_ROOT) / f"clip_{tag}"


def load_ratio_result(tag: str):
    """Return the saved result dict for a finished ratio, or None if not complete."""
    p = ratio_output_dir(tag) / "result.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def save_ratio_result(tag: str, res: dict):
    out = ratio_output_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "result.json", "w") as f:
        json.dump(res, f)


def collect_all_results():
    """Rebuild the results dict from on-disk markers (for re-plotting after a crash)."""
    ret = {}
    for ratio in ROI_RATIOS:
        tag = f"ratio_{int(ratio * 100)}"
        res = load_ratio_result(tag)
        if res is not None:
            ret[ratio] = res
    return ret


# -------------------------------------------------
# MAIN
# -------------------------------------------------


def main():
    ret = {}
    logger = setup_multi_loops_logger()

    print("Loading eye-center metadata (full glint360k)...")
    metadata = load_eye_center_metadata(EYE_CENTER_METADATA_PATH)
    print(f"Loaded {len(metadata)} metadata entries.")

    for ratio in ROI_RATIOS:
        tag = f"ratio_{int(ratio * 100)}"

        cached = load_ratio_result(tag)
        if cached is not None:
            print(f"⏭️  {tag} already complete (resume): {cached}, skipping.")
            logger.info(f"Resume: {tag} already complete: {cached}, skipping.")
            ret[ratio] = cached
            continue

        save_visual_inspection_grid(metadata, ratio, tag)

        if TRAINING_FLAG:
            args = read_and_update_variant_config(
                tag, ratio=ratio, metadata=metadata, exp_name_prefix="clip",
            )

            print(f"🎯 Training ArcFace on {tag}")
            logger.info(f"Training ArcFace on {tag}")
            ret[ratio] = run_training(args=args)

            save_ratio_result(tag, ret[ratio])

            print(f"✅ Completed training for {tag} with results: {ret[ratio]}")
            logger.info(f"Completed training for {tag} with results: {ret[ratio]}")

    plot_results(ret)
    plot_accuracy_comparison(ret, ACCURACY_PLOT_PATH)


if __name__ == "__main__":
    main()
