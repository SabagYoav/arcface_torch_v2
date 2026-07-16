"""
Rebuild LFW partial_15 to MATCH the on-disk variants_dataset/ratio_15 band
geometry (decoded empirically in decode_variants_geometry.py):
    y1 = round(pupil_y + Y1_OFF),  y2 = round(pupil_y + Y2_OFF)   (full width)
where pupil_y is the detected eye-center already cached in landmarks.json.
Both glint and LFW are aligned to the same 112 template, so pupil_y is on the
same scale => the LFW partials land in the student's training distribution.
"""
import json
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm

ROOT = Path("/media/yoav/Yoav/datasets/benchmarks/lfw")
FULL = ROOT / "full"
OUT = ROOT / "partial_15"          # overwrite the mismatched version
LM = json.load(open(ROOT / "landmarks.json"))
GEO = json.load(open(ROOT / "variants_geometry.json"))
Y1_OFF, Y2_OFF = GEO["y1_off"], GEO["y2_off"]
print(f"Using offsets y1=pupil_y+{Y1_OFF:.2f}  y2=pupil_y+{Y2_OFF:.2f}")

WORK = Path("work_dirs/benchmarks/lfw")
grid = []
measured_y1, measured_y2 = [], []

for key, meta in tqdm(LM.items(), desc="rebuild partial_15"):
    py = meta["center_y"]
    src = FULL / key
    bgr = cv2.imread(str(src))
    if bgr is None:
        continue
    H, W = bgr.shape[:2]
    y1 = int(np.clip(round(py + Y1_OFF), 0, H))
    y2 = int(np.clip(round(py + Y2_OFF), 0, H))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[y1:y2, :] = bgr[y1:y2, :]
    out = OUT / key
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    measured_y1.append(y1); measured_y2.append(y2)
    if len(grid) < 16:
        grid.append(canvas)

my1, my2 = np.array(measured_y1), np.array(measured_y2)
print(f"Rebuilt {len(my1)} partials.  band y1={my1.mean():.1f}±{my1.std():.1f}  "
      f"y2={my2.mean():.1f}±{my2.std():.1f}  height={ (my2-my1).mean():.1f}")
print("target (variants ratio_15): y1~34.0  y2~78.6  height~45.6")

if len(grid) >= 16:
    rows = [np.hstack(grid[i:i+4]) for i in range(0, 16, 4)]
    cv2.imwrite(str(WORK / "lfw_partial_15_matched_grid.jpg"), np.vstack(rows))
    print(f"grid -> {WORK/'lfw_partial_15_matched_grid.jpg'}")
