"""
Generate the 15% ROI *partial-face* domain for LFW.

Input : LFW already ArcFace-aligned to 112x112 (the full-face / teacher domain).
Step  : detect 5-pt landmarks (insightface SCRFD) -> eye-center y.
Crop  : replicate data_scratches/build_roi_datasets.py EXACTLY so the partial
        faces are in-distribution for the ratio_15 ViT-S student:
            roi_h = int(H * ROI_RATIO)   (ROI_RATIO = 0.15)
            roi_w = int(W * 1.0)         (full width)
            cy    = eye_center_y - SHIFT_Y   (SHIFT_Y = 15)
            band  = img[y1:y2, :]  pasted onto a black 112x112 canvas
Output: partial_15/<Name>/<img>.jpg  + landmark cache + preview grids.

If the detector finds no face on the 112px crop we fall back to the ArcFace
template eye-center (canonical y=51.69) — the images are already aligned so
this is the correct eye row by construction. Fallback count is logged (no
silent truncation).
"""
import os, sys, json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

_this = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this, "..", ".."))

# ----- match build_roi_datasets.py -----
ROI_RATIO = 0.15
ROI_WIDTH_RATIO = 1.0
SHIFT_Y = 15
# ArcFace 112 template eye centers: left (38.29,51.69), right (73.53,51.69)
TEMPLATE_EYE_Y = 51.69

ROOT = Path("/media/yoav/Yoav/datasets/benchmarks/lfw")
FULL_DIR = ROOT / "full"
PARTIAL_DIR = ROOT / "partial_15"
LM_CACHE = ROOT / "landmarks.json"
WORK_DIR = Path("work_dirs/benchmarks/lfw")
WORK_DIR.mkdir(parents=True, exist_ok=True)


def build_detector():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(160, 160))
    return app


def eye_center_y(app, bgr):
    """Return (center_y, detected) using the mean y of the two eye landmarks."""
    faces = app.get(bgr)
    if not faces:
        return TEMPLATE_EYE_Y, False
    # most central / largest face
    H, W = bgr.shape[:2]
    c = np.array([W / 2, H / 2])
    def score(f):
        bb = f.bbox
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        return area - 2.0 * np.hypot(cx - c[0], cy - c[1])
    f = max(faces, key=score)
    kps = f.kps  # [left_eye, right_eye, nose, left_mouth, right_mouth]
    return float((kps[0][1] + kps[1][1]) / 2.0), True


def make_partial(bgr, center_y):
    """Exact port of build_roi_datasets.write_split cropping."""
    H, W = bgr.shape[:2]
    roi_h = int(H * ROI_RATIO)
    roi_w = int(W * ROI_WIDTH_RATIO)
    cx = W // 2
    cy = int(round(center_y)) - SHIFT_Y
    y1 = max(0, cy - roi_h // 2)
    y2 = min(H, y1 + roi_h)
    x1 = max(0, cx - roi_w // 2)
    x2 = min(W, x1 + roi_w)
    roi = bgr[y1:y2, x1:x2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    if roi.size:
        canvas[y1:y2, x1:x2] = roi
    return canvas


def main():
    app = build_detector()
    imgs = sorted(FULL_DIR.glob("*/*.jpg"))
    print(f"Processing {len(imgs)} aligned LFW images -> partial_15")

    lm = {}
    n_fallback = 0
    grid_full, grid_part = [], []

    for p in tqdm(imgs, desc="crop partial_15"):
        out = PARTIAL_DIR / p.parent.name / p.name
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        cy, detected = eye_center_y(app, bgr)
        if not detected:
            n_fallback += 1
        lm[f"{p.parent.name}/{p.name}"] = {"center_y": cy, "detected": detected}
        part = make_partial(bgr, cy)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), part)
        if len(grid_full) < 16:
            grid_full.append(bgr); grid_part.append(part)

    with open(LM_CACHE, "w") as f:
        json.dump(lm, f)

    # preview grids (4x4) mirroring the training-time batch grids
    def save_grid(imgs_list, path):
        rows = [np.hstack(imgs_list[i:i+4]) for i in range(0, 16, 4)]
        cv2.imwrite(str(path), np.vstack(rows))
    if len(grid_full) >= 16:
        save_grid(grid_full, WORK_DIR / "lfw_full_grid.jpg")
        save_grid(grid_part, WORK_DIR / "lfw_partial_15_grid.jpg")

    print(f"Done. detected={len(imgs)-n_fallback}  fallback(template)={n_fallback}")
    print(f"partial_15 -> {PARTIAL_DIR}")
    print(f"grids -> {WORK_DIR}")


if __name__ == "__main__":
    main()
