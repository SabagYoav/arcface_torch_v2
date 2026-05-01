import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
IMAGE_DIR = "/media/yoav/Yoav/datasets/glint360k/imageFolder_split_single_narrow_eye/val"
OUT_JSON  = "/media/yoav/Yoav/datasets/glint360k/center_of_eyes_points_val.json"

results = {}

for id_dir in tqdm(sorted(Path(IMAGE_DIR).glob("*"))):
    for img_path in sorted(id_dir.glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find non-zero pixels
        ys, xs = np.where(gray > 0)

        if len(ys) == 0:
            results[img_path.name] = {
                "detected": False,
                "center_y": None
            }
            continue

        y_min = int(ys.min())
        y_max = int(ys.max())
        center_y = float((y_min + y_max) / 2)

        results[img_path.name] = {
            "detected": True,
            "center_y": center_y
        }

# -------------------------
# Save JSON
# -------------------------
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved → {OUT_JSON}")
