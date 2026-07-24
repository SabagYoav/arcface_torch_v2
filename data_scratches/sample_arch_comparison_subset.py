"""Sample a random identity subset from the full glint360k for the standard
(non-distillation) multi-architecture training comparison — full-face, no ROI
cropping. Symlinks whole identity directories (instant, disk-free), mirroring
scripts/make_fullface_subset.py's pattern, but sampling randomly from the flat
360k-identity pool (which has no existing train/val split) instead of picking
the first N from an already-split tree.
"""
import os
import json
import random
import shutil
from pathlib import Path

SRC = Path("/DATA/glint360k_download/glint360k_images")
DST = Path("/DATA/glint360k_arch_comparison")
IMG_EXTS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES = 5
N_TRAIN = 36000
N_VAL = 9000
SEED = 42


def count_images(d: Path) -> int:
    return sum(1 for p in os.scandir(d) if Path(p.name).suffix.lower() in IMG_EXTS)


def main():
    all_ids = sorted(e.name for e in os.scandir(SRC) if e.is_dir())
    print(f"total identities available: {len(all_ids)}")

    rng = random.Random(SEED)
    rng.shuffle(all_ids)

    selected = []
    for id_name in all_ids:
        if len(selected) >= N_TRAIN + N_VAL:
            break
        if count_images(SRC / id_name) >= MIN_IMAGES:
            selected.append(id_name)

    print(f"selected {len(selected)} identities meeting MIN_IMAGES={MIN_IMAGES}")
    assert len(selected) >= N_TRAIN + N_VAL, (
        f"only found {len(selected)} qualifying identities, need {N_TRAIN + N_VAL}"
    )

    train_ids = selected[:N_TRAIN]
    val_ids = selected[N_TRAIN:N_TRAIN + N_VAL]

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        dst_split = DST / split
        if dst_split.exists():
            shutil.rmtree(dst_split)
        dst_split.mkdir(parents=True)
        for id_name in ids:
            os.symlink(SRC / id_name, dst_split / id_name)
        print(f"{split}: linked {len(ids)} identities -> {dst_split}")

    manifest = {
        "seed": SEED, "min_images": MIN_IMAGES,
        "n_train": len(train_ids), "n_val": len(val_ids),
        "train_ids": train_ids, "val_ids": val_ids,
    }
    DST.mkdir(parents=True, exist_ok=True)
    with open(DST / "sample_manifest.json", "w") as f:
        json.dump(manifest, f)
    print(f"manifest -> {DST / 'sample_manifest.json'}")


if __name__ == "__main__":
    main()
