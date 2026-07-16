"""
Build an LFW partial-face set at a LITERAL ratio of face height, centered on the
detected eye landmarks. Clean, well-defined geometry (no dependence on the messy
on-disk variants_dataset):

    roi_h = round(RATIO * 112)          # e.g. 0.20 -> 22 px
    cy    = detected eye-center y        # from landmarks.json
    band  = full[cy-roi_h//2 : ..., :]   # full width, pasted on a black canvas

Usage:  build_lfw_partial.py 0.20
"""
import sys, json
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm

RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
TAG = f"partial_{int(round(RATIO*100))}"

ROOT = Path("/media/yoav/Yoav/datasets/benchmarks/lfw")
FULL = ROOT / "full"
OUT = ROOT / TAG
LM = json.load(open(ROOT / "landmarks.json"))
WORK = Path("work_dirs/benchmarks/lfw"); WORK.mkdir(parents=True, exist_ok=True)

y1s, y2s, grid = [], [], []
for key, meta in tqdm(LM.items(), desc=f"build {TAG}"):
    py = meta["center_y"]
    bgr = cv2.imread(str(FULL / key))
    if bgr is None:
        continue
    H, W = bgr.shape[:2]
    roi_h = int(round(RATIO * H))
    cy = int(round(py))
    y1 = max(0, cy - roi_h // 2)
    y2 = min(H, y1 + roi_h)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[y1:y2, :] = bgr[y1:y2, :]
    out = OUT / key
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    y1s.append(y1); y2s.append(y2)
    if len(grid) < 16:
        grid.append(canvas)

y1s, y2s = np.array(y1s), np.array(y2s)
print(f"\n{TAG}: RATIO={RATIO}  roi_h={int(round(RATIO*112))}px  "
      f"band y1={y1s.mean():.1f}±{y1s.std():.1f}  y2={y2s.mean():.1f}±{y2s.std():.1f}  "
      f"height={(y2s-y1s).mean():.1f}px  n={len(y1s)}")

if len(grid) >= 16:
    rows = [np.hstack(grid[i:i+4]) for i in range(0, 16, 4)]
    p = WORK / f"lfw_{TAG}_grid.jpg"
    cv2.imwrite(str(p), np.vstack(rows))
    print(f"grid -> {p}")

# glint reference grid at the same ratio, if present
gl = Path(f"/media/yoav/Yoav/datasets/glint360k/ROIs/ratio_{int(round(RATIO*100))}/test")
if gl.exists():
    import glob
    gs = sorted(glob.glob(str(gl / "*/*.jpg")))[:16]
    if len(gs) >= 16:
        imgs = [cv2.imread(x) for x in gs]
        rows = [np.hstack(imgs[i:i+4]) for i in range(0, 16, 4)]
        p = WORK / f"glint_ratio_{int(round(RATIO*100))}_grid.jpg"
        cv2.imwrite(str(p), np.vstack(rows))
        print(f"glint reference grid -> {p}")
