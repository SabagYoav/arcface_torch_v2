import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch

from train_v4_clip import train
from utils.utils_config import get_config
from dataset import load_eye_center_metadata

TRAIN_DIR = Path("/DATA/glint360k_arch_comparison/train")
VAL_DIR = Path("/DATA/glint360k_arch_comparison/val")
MANIFEST_PATH = Path("/DATA/glint360k_arch_comparison/sample_manifest.json")
FULL_METADATA_PATH = Path("/DATA/glint360k_download/eye_center_metadata/center_eyes_metadata.json")
TRAIN_METADATA_PATH = Path("/DATA/glint360k_arch_comparison/train_eye_center_metadata.json")
VAL_METADATA_PATH = Path("/DATA/glint360k_arch_comparison/val_eye_center_metadata.json")
TEACHER_MODEL_PATH = Path("16backbone.pth")

BASE_CONFIG = "configs/experiment_arch_comparison_clip.py"
EXP_NAME = os.environ.get("EXP_NAME", "exp_arch_comparison")
EXP_ROOT = f"work_dirs/{EXP_NAME}"

# (architecture name matching backbones.get_model(), batch_size)
ARCHITECTURES = [
    ("r50", 256),
    ("r100", 256),
    ("vit_b16", 128),
    ("swin_tiny", 128),
    ("mobilevit_s", 128),
]
ROI_RATIOS = [0.15, 0.2, 0.4, 0.6, 0.8, 1.0]


def build_scoped_metadata():
    """Reuse the full glint360k eye-center metadata (already computed for the
    main sweep) filtered down to just this comparison's sampled identities —
    no new buffalo_l detection needed."""
    if TRAIN_METADATA_PATH.exists() and VAL_METADATA_PATH.exists():
        print("Scoped metadata already built, skipping.")
        return

    print("Loading full glint360k eye-center metadata (one-time)...")
    full_metadata = load_eye_center_metadata(FULL_METADATA_PATH)
    manifest = json.load(open(MANIFEST_PATH))
    train_ids = set(manifest["train_ids"])
    val_ids = set(manifest["val_ids"])

    train_meta, val_meta = {}, {}
    for k, v in full_metadata.items():
        id_name = k.split("/", 1)[0]
        if id_name in train_ids:
            train_meta[k] = v
        elif id_name in val_ids:
            val_meta[k] = v

    with open(TRAIN_METADATA_PATH, "w") as f:
        json.dump(train_meta, f)
    with open(VAL_METADATA_PATH, "w") as f:
        json.dump(val_meta, f)
    print(f"Scoped metadata: {len(train_meta)} train entries, {len(val_meta)} val entries")


def tag_for(arch: str, ratio: float) -> str:
    return f"{arch}_ratio_{int(round(ratio * 100))}"


def output_dir(arch: str, ratio: float) -> Path:
    return Path(EXP_ROOT) / tag_for(arch, ratio)


def load_result(arch: str, ratio: float):
    p = output_dir(arch, ratio) / "result.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def save_result(arch: str, ratio: float, res: dict):
    out = output_dir(arch, ratio)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "result.json", "w") as f:
        json.dump(res, f)


def write_variants_config(dst_root: Path, cfg: dict):
    with open(dst_root, "w") as f:
        f.write("# Auto-generated config file\n")
        f.write("from easydict import EasyDict as edict\n\n")
        f.write("config = edict()\n")
        for key, value in cfg.items():
            f.write(f"config.{key} = {repr(value)}\n")


def build_config(arch: str, batch_size: int, ratio: float, num_classes: int, num_image: int):
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", default=BASE_CONFIG, type=str)
    args = parser.parse_args([])

    cfg = get_config(args.config)
    cfg.network = arch
    cfg.batch_size = batch_size
    cfg.use_onthefly_dataset = True
    cfg.root_dir = str(TRAIN_DIR)
    cfg.eye_center_metadata_path = str(TRAIN_METADATA_PATH)
    cfg.roi_ratio = ratio
    cfg.teacher_model_path = str(TEACHER_MODEL_PATH)
    cfg.glint_val_root = str(VAL_DIR)
    cfg.glint_val_metadata_path = str(VAL_METADATA_PATH)
    cfg.output = str(output_dir(arch, ratio))
    cfg.num_classes = num_classes
    cfg.num_image = num_image

    tag = tag_for(arch, ratio)
    cfg_path = Path(f"configs/variants_arch_{tag}.py")
    write_variants_config(cfg_path, cfg)

    args.config = str(cfg_path)
    return args


def plot_comparison(results: dict, filename: str = None):
    if not results:
        print("No results to plot.")
        return
    filename = filename or os.path.join(EXP_ROOT, "arch_comparison.png")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    by_arch = {}
    for tag, res in results.items():
        arch, _, ratio_tag = tag.rpartition("_ratio_")
        by_arch.setdefault(arch, []).append((int(ratio_tag) / 100.0, res.get("best_acc", float("nan"))))

    plt.figure(figsize=(8, 5))
    for arch, points in by_arch.items():
        points.sort()
        xs = [int(round(r * 100)) for r, _ in points]
        ys = [a for _, a in points]
        plt.plot(xs, ys, marker="o", label=arch)

    plt.xlabel("ROI ratio (% of face height visible)")
    plt.ylabel("Best verification accuracy (LFW cross_partial_vs_full)")
    plt.title("Architecture comparison — glint360k subset (36k/9k identities)")
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved comparison plot -> {filename}")


def main():
    build_scoped_metadata()

    train_metadata = load_eye_center_metadata(TRAIN_METADATA_PATH)
    detected = [k for k, v in train_metadata.items() if v.get("detected", False)]
    num_classes = len({k.split("/", 1)[0] for k in detected})
    num_image = len(detected)
    print(f"Training subset: {num_classes} identities, {num_image} images")
    del train_metadata, detected

    results = {}
    for arch, batch_size in ARCHITECTURES:
        for ratio in ROI_RATIOS:
            tag = tag_for(arch, ratio)
            cached = load_result(arch, ratio)
            if cached is not None:
                print(f"Skipping {tag} (already complete): {cached}")
                results[tag] = cached
                continue

            args = build_config(arch, batch_size, ratio, num_classes, num_image)
            print(f"Training {tag}...")
            res = train(args)
            save_result(arch, ratio, res)
            results[tag] = res
            print(f"Completed {tag}: {res}")

    plot_comparison(results)


if __name__ == "__main__":
    main()
