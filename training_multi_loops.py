import os
import cv2
import json
import torch
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from train_v4_clip import train
from utils.utils_config import get_config

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
FULLFACE_ROOT = Path("/media/yoav/Yoav/datasets/glint360k/ROIs/ratio_100")  # quick-run subset (symlinked IDs)
CENTER_Y_JSON = Path("/media/yoav/Yoav/datasets/glint360k")

# ROI_ROOT = Path("/datasets/roi_variants")
TRAIN_SCRIPT = Path("train_v4_clip.py")

# Experiment config + output namespace. Override per experiment via env vars, e.g.
#   EXP_NAME=exp_r50_vs_r50_clip BASE_CONFIG=configs/experiment_r50_vs_r50_clip.py python training_multi_loops_claude.py
# so each experiment writes to its own work_dirs/<EXP_NAME>/ folder (and its
# result.json markers only block reruns of that same experiment).
BASE_CONFIG = 'configs/experiment_r50_vs_vit_clip.py' #os.environ.get("BASE_CONFIG", "configs/experiment_r50_vs_r50_clip.py")
EXP_NAME = os.environ.get("EXP_NAME", "exp_clip_r50_vs_vit")  # output namespace for this experiment (work_dirs/<EXP_NAME>/)

ROI_RATIOS = [1.0, 0.6, 0.4, 0.3, 0.2, 0.15]
ROI_WIDTH_RATIO = 1.0   # relative to face width
IMG_EXTS = [".jpg", ".jpeg", ".png"]

VARIANTS_DATASET_ROOT = Path("/media/yoav/Yoav/datasets/variants_dataset")

MARGIN = 5

ACCURACY_PLOT_PATH = os.path.join("work_dirs", EXP_NAME, "fullset_accuracy_comparison.png")

TRAINING_FLAG = True
# GPU_MEM_FRACTION = 0.9   

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

# -------------------------------------------------
# ROI writer (uses your earlier logic)
# -------------------------------------------------
def remove_small_id_folders(root: Path, min_images: int = 2):
    for id_dir in root.iterdir():
        if not id_dir.is_dir():
            continue

        num_images = sum(
            1 for p in id_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS
        )

        if num_images < min_images:
            shutil.rmtree(id_dir)
            print(f"🗑️ Removed {id_dir.name} (only {num_images} images)")

def write_split(
    src_root: Path,
    center_y_json: Path,
    dst_root: Path,
    roi_ratio: float,
    roi_w_ratio: float
):
    with open(center_y_json, "r") as f:
        center_data = json.load(f)

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    for id_dir in src_root.iterdir():
        if not id_dir.is_dir():
            continue

        out_id_dir = dst_root / id_dir.name
        out_id_dir.mkdir(parents=True)

        for img_path in id_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            meta = center_data.get(img_path.name)
            if meta is None or not meta.get("detected", False):
                continue

            center_y = int(meta["center_y"])

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            H, W = img.shape[:2]
            roi_h = int(H * roi_ratio)
            roi_w = int(W * roi_w_ratio)

            cx = W // 2
            cy = center_y

            y1 = max(0, (cy - roi_h // 2)+MARGIN )
            y2 = min(H, (y1 + roi_h)+MARGIN )
            x1 = max(0, cx - roi_w // 2)
            x2 = min(W, x1 + roi_w)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            canvas[y1:y2, x1:x2] = roi

            cv2.imwrite(str(out_id_dir / img_path.name), canvas)

    # remove empty or too small folders
    remove_small_id_folders(dst_root, min_images=2)


def plot_metric_vs_variant(results: dict, metric: str, filename: str):
    # sort by variant (optional but recommended)
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
    import os
    os.makedirs(out_dir, exist_ok=True)

    # assume all entries share same metric keys
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
    plt.ylabel("Best verification accuracy (val)")
    plt.title("R50 student vs R50 teacher — CLIP distillation accuracy vs ROI")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()   # full face (100) on the left, eyes-only (15) on the right
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved accuracy comparison plot to {filename}")


def read_and_update_variant_config(tag:str, ff_dir:Path, pf_dir:Path, exp_name_prefix = None):
    train_ff_dir, val_ff_dir = str(ff_dir / "train"), str(ff_dir / "val")
    train_pf_dir, val_pf_dir  = str(pf_dir / "train"), str(pf_dir / "val")


    val_targets = [val_ff_dir, val_pf_dir ]
    train_targets = [train_ff_dir,train_pf_dir]
    exp_name = f"{exp_name_prefix}_{tag}"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch"
    )
    parser.add_argument(
        "--config",
        default=BASE_CONFIG,
        type=str,
    )
    args = parser.parse_args([])

    cfg = get_config(args.config)
    # cfg.rec = str(train_dir)
    cfg.root_ff = train_ff_dir
    cfg.root_pf = train_pf_dir
    cfg.val_targets = val_targets
    cfg.train_targets = train_targets
    cfg.output = str(ratio_output_dir(tag))   # work_dirs/exp_r50_vs_r50/clip_<tag>
    cfg.batch_grid_tag = exp_name


    cfg.num_classes = sum(
        1 for p in Path(train_pf_dir).iterdir()
        if p.is_dir()
    )

    cfg.num_image = sum(
        sum(1 for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        for cls_dir in Path(train_pf_dir).iterdir()
        if cls_dir.is_dir()
    )

    # cfg_path = Path(f"configs/variants_config.py")
    cfg_path = Path(f"configs/variants_config_{tag}.py")
    write_variants_config(dst_root=cfg_path, cfg=cfg)

    args.config = str(cfg_path)
    return args


def run_training(args):
    results_dict = train(args)
    return results_dict


def save_split_sample(dst_root: Path, split: str):
    #TODO: implement save grid of 5X5 images for visual verification
    pass


def write_variants_config(dst_root: Path, cfg: dict):
    with open(dst_root, "w") as f:
        f.write("# Auto-generated config file\n")
        f.write("from easydict import EasyDict as edict\n\n")
        f.write("config = edict()\n")

        for key, value in cfg.items():
            f.write(f"config.{key} = {repr(value)}\n")


def write_fingerprint(dst_root: Path, filename: str, tag: str = None):
    fingerprint_path = dst_root / filename
    with open(fingerprint_path, "w") as f:
        f.write(f"{tag}")


def read_fingerprint(dst_root: Path, filename: str):
    fingerprint_path = dst_root / filename
    if not fingerprint_path.exists():
        return None
    with open(fingerprint_path, "r") as f:
        return f.read().strip()


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


def write_variant_dataset(ratio):
    # Each ratio gets its own tag dir (VARIANTS_DATASET_ROOT/ratio_XX/{train,val,test}).
    # Skip rebuilding if a per-tag fingerprint already marks it complete.
    tag = f"ratio_{int(ratio * 100)}"
    tag_root = VARIANTS_DATASET_ROOT / tag

    fingerprint = read_fingerprint(dst_root=tag_root, filename="fingerprint.txt")
    if fingerprint == tag:
        print(f"⏭️  ROI dataset {tag} already built, skipping.")
        return

    for split in ["train", "val", "test"]:
        print(f"📂 Writing split: {split} with ROI ratio: {ratio}")
        write_split(
            src_root=FULLFACE_ROOT / split,
            center_y_json=CENTER_Y_JSON / f"center_of_eyes_points_{split}.json",
            dst_root=tag_root / split,
            roi_ratio=ratio,
            roi_w_ratio=ROI_WIDTH_RATIO
        )
    write_fingerprint(dst_root=tag_root, filename="fingerprint.txt", tag=tag)

# -------------------------------------------------
# MAIN
# -------------------------------------------------


def main():
    ret = {}

    # Cap this process to <= 0.5 of the GPU memory (hard limit; OOMs instead of exceeding).
    # if torch.cuda.is_available():
    #     torch.cuda.set_per_process_memory_fraction(GPU_MEM_FRACTION, 0)

    ## define logger
    logger = setup_multi_loops_logger()
    log_path = "training_multi_loops_log.txt"

    ## iterate over variants (ROIs)
    for ratio in ROI_RATIOS:
        tag = f"ratio_{int(ratio * 100)}"

        ## resume: skip ratios that already finished (have a result.json marker)
        cached = load_ratio_result(tag)
        if cached is not None:
            print(f"⏭️  {tag} already complete (resume): {cached}, skipping.")
            logger.info(f"Resume: {tag} already complete: {cached}, skipping.")
            ret[ratio] = cached
            continue

        ## create partial face data variant
        print(f"\n📦 Creating ROI dataset: {tag}")
        logger.info(f"Creating ROI dataset: {tag}")
        write_variant_dataset(ratio)

        if TRAINING_FLAG == True:

            ## read and update variant config file
            #TODO: set the variables in an env file
            args = read_and_update_variant_config(
                tag,
                ff_dir=FULLFACE_ROOT,
                pf_dir=VARIANTS_DATASET_ROOT / tag,   # read from the same tag dir we just wrote
                exp_name_prefix="clip",
            )

            ## run training loop for the variant (train_v4 resumes mid-ratio from its own checkpoint)
            print(f"🎯 Training ArcFace on {tag}")
            logger.info(f"Training ArcFace on {tag}")
            ret[ratio] = run_training(args=args)

            ## persist per-ratio result so a later crash won't redo this ratio
            save_ratio_result(tag, ret[ratio])

            ## log results
            print(f"✅ Completed training for {tag} with results: {ret[ratio]}")
            logger.info(f"Completed training for {tag} with results: {ret[ratio]}")

    plot_results(ret)
    plot_accuracy_comparison(ret, ACCURACY_PLOT_PATH)

if __name__ == "__main__":
    main()

    # TODO: add roc curve etc
