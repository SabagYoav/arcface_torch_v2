"""Create a small symlinked subset of the fullface dataset for quick experiments.

Picks the first N identity folders (sorted) that have >= MIN_IMAGES images and
symlinks the *directories* into a parallel `fullface_subset/{train,val,test}` tree.
Symlinking dirs keeps it instant and disk-free; downstream code (cv2 / Path.glob)
follows the links transparently.
"""
import os
from pathlib import Path

SRC = Path("/media/yoav/Yoav/datasets/glint360k/imageFolder_split_fullface")
DST = Path("/media/yoav/Yoav/datasets/glint360k/fullface_subset")
IMG_EXTS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES = 3
N_PER_SPLIT = {"train": 2500, "val": 500, "test": 50}


def count_images(d: Path) -> int:
    return sum(1 for p in os.scandir(d) if Path(p.name).suffix.lower() in IMG_EXTS)


def build_split(split: str, n: int):
    src_split = SRC / split
    dst_split = DST / split
    if dst_split.exists():
        import shutil
        shutil.rmtree(dst_split)
    dst_split.mkdir(parents=True)

    picked = 0
    for entry in sorted(os.scandir(src_split), key=lambda e: e.name):
        if picked >= n:
            break
        if not entry.is_dir():
            continue
        if not entry.name.isdigit():
            continue
        if count_images(Path(entry.path)) < MIN_IMAGES:
            continue
        os.symlink(entry.path, dst_split / entry.name)
        picked += 1
    print(f"{split}: linked {picked} identities -> {dst_split}")
    return picked


if __name__ == "__main__":
    for split, n in N_PER_SPLIT.items():
        build_split(split, n)
    print("done")
