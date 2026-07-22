"""
LFW loading for the cross-domain (teacher-full vs student-partial) verification
used by ClipVerification. Decodes the standard insightface lfw.bin directly
(pickle of (bins, issame_list), bins = JPEG-encoded bytes) via cv2 — no mxnet
dependency needed (unlike eval/verification.py's load_bin, which does).

Pair/fold structure matches the standard LFW protocol used throughout this
codebase (see eval/verification.py's LFold): 6000 pairs, 10 folds of 600
consecutive pairs each, image 2i/2i+1 forming pair i — no separate pairs.txt
needed, it's implicit in the bin's fixed ordering.
"""
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

LFW_BIN_PATH = Path("/DATA/glint360k_download/glint360k_extracted/glint360k/lfw.bin")
EYE_CENTER_CACHE = Path("/DATA/glint360k_download/glint360k_extracted/glint360k/lfw_eye_centers.json")
TEMPLATE_EYE_Y = 51.69  # ArcFace 112 template canonical eye row, same fallback as glint metadata
N_FOLDS = 10


def decode_lfw_bin(path=LFW_BIN_PATH):
    """Returns (images, issame_list). images[i] is a 112x112x3 BGR uint8 array;
    images[2i]/images[2i+1] form pair i, labeled issame_list[i]."""
    with open(path, "rb") as f:
        try:
            bins, issame_list = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            bins, issame_list = pickle.load(f, encoding="bytes")
    images = [cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR) for b in bins]
    return images, list(issame_list)


def pair_folds(n_pairs, n_folds=N_FOLDS):
    """Sequential-block fold assignment, matching eval/verification.py's
    non-shuffled KFold over consecutive pair indices."""
    fold_size = n_pairs // n_folds
    return np.array([min(i // fold_size, n_folds - 1) for i in range(n_pairs)])


def compute_or_load_eye_centers(images, app, cache_path=EYE_CENTER_CACHE):
    """List of {"center_y": float, "detected": bool}, aligned with `images`
    index. Cached to disk (LFW is small — 12k images — so this is a one-off
    cost, unlike the full glint360k metadata job)."""
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) == len(images):
            return cached

    from dataset import crop_roi  # local import to avoid a hard dep at module load

    results = []
    for bgr in images:
        faces = app.get(bgr)
        if not faces:
            results.append({"center_y": TEMPLATE_EYE_Y, "detected": False})
            continue
        H, W = bgr.shape[:2]
        c = np.array([W / 2, H / 2])

        def score(f):
            bb = f.bbox
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            return area - 2.0 * np.hypot(cx - c[0], cy - c[1])

        f = max(faces, key=score)
        cy = float((f.kps[0][1] + f.kps[1][1]) / 2.0)
        results.append({"center_y": cy, "detected": True})

    with open(cache_path, "w") as f:
        json.dump(results, f)
    return results
