"""
Empirically decode the ratio_15 partial-face band geometry used to build
variants_dataset (the historical generator params are lost). We pair, per image:
  - pupil_y  : detected eye-center y on the FULL face (glint ROIs/ratio_100)
  - (y1,y2)  : the actual non-zero band rows in variants_dataset/ratio_15
and regress band edges on pupil_y. The fitted rule is applied to LFW using the
SAME pupil-based definition on both sides (see build_lfw_partial_matched.py).
"""
import glob, json
from pathlib import Path
import numpy as np, cv2
from insightface.app import FaceAnalysis

FULL = Path("/media/yoav/Yoav/datasets/glint360k/ROIs/ratio_100/test")
PART = Path("/media/yoav/Yoav/datasets/variants_dataset/ratio_15/test")
N = 500

app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(160, 160))


def band(p):
    im = cv2.imread(str(p)); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ys = np.where(g.sum(1) > 0)[0]
    return (int(ys.min()), int(ys.max())) if len(ys) else None


def pupil_y(p):
    im = cv2.imread(str(p)); f = app.get(im)
    if not f:
        return None
    f = max(f, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    return float((f.kps[0][1] + f.kps[1][1]) / 2.0)


rows = []
part_imgs = glob.glob(str(PART / "*/*.jpg"))
np.random.default_rng(0).shuffle(part_imgs)
for pp in part_imgs:
    pp = Path(pp)
    fp = FULL / pp.parent.name / pp.name
    if not fp.exists():
        continue
    b = band(pp); py = pupil_y(fp)
    if b is None or py is None:
        continue
    rows.append((py, b[0], b[1]))
    if len(rows) >= N:
        break

rows = np.array(rows)
py, y1, y2 = rows[:, 0], rows[:, 1], rows[:, 2]
print(f"n={len(rows)}  pupil_y mean={py.mean():.2f}  y1 mean={y1.mean():.2f}  y2 mean={y2.mean():.2f}  height={ (y2-y1+1).mean():.2f}")

for name, y in [("y1", y1), ("y2", y2)]:
    a, b0 = np.polyfit(py, y, 1)
    pred = a * py + b0
    ss = 1 - ((y - pred) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-9)
    print(f"{name} = {a:.4f} * pupil_y + {b0:.4f}   R^2={ss:.3f}   resid_sd={np.std(y-pred):.2f}")

# Also report the simple constant-offset model (band tracks pupil rigidly)
d1 = (y1 - py); d2 = (y2 - py)
print(f"offset model: y1 = pupil_y + ({d1.mean():.2f} ± {d1.std():.2f}),  y2 = pupil_y + ({d2.mean():.2f} ± {d2.std():.2f})")
json.dump({"y1_off": float(d1.mean()), "y2_off": float(d2.mean()),
           "y1_lin": np.polyfit(py, y1, 1).tolist(), "y2_lin": np.polyfit(py, y2, 1).tolist(),
           "pupil_y_mean_glint": float(py.mean())},
          open("/media/yoav/Yoav/datasets/benchmarks/lfw/variants_geometry.json", "w"), indent=2)
print("saved variants_geometry.json")
