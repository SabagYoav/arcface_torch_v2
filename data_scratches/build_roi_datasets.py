import json
import shutil
from pathlib import Path
import cv2
import numpy as np

ROI_RATIOS = [0.4]  # List of ROI height ratios to create datasets for
ROI_WIDTH_RATIO = 1.0  # Relative to face width
IMG_EXTS = [".jpg", ".jpeg", ".png"]  # Supported image extensions  
SRC_DATASET_ROOT = Path("/datasets/glint360k/imageFolder_split_fullface")  # Root directory for full-face images
CENTER_Y_JSON = Path("/datasets/glint360k")  # Directory containing center_y JSON files
DST_DATASET_ROOT = Path("/datasets/glint360k/ROIs")  # Root directory for saving ROI datasets
SHIFT_Y = 15  # Margin in pixels to add around the ROI

def write_split(
    src_root: Path,
    center_y_json: Path,
    dst_root: Path,
    roi_ratio: float,
    roi_w_ratio: float
):
    # Load center_y data from json file
    with open(center_y_json, "r") as f:
        center_data = json.load(f)
    # Prepare destination base(train/test/val level) directory, create if not exists, or clear if already exists
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
  # iterate over each ID directory in the source dataset
    for id_dir in src_root.iterdir():
        if not id_dir.is_dir():
            continue
        # Create corresponding ID directory in the destination dataset
        # out_id_dir = dst_root / id_dir.name
        # out_id_dir.mkdir(parents=True)

        for img_path in id_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            meta = center_data.get(img_path.name)
            if meta is None or not meta.get("detected", False):
                continue
            
            # Create corresponding ID directory in the destination dataset
            out_id_dir = dst_root / id_dir.name
            out_id_dir.mkdir(parents=True, exist_ok=True)

            center_y = int(meta["center_y"])

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            H, W = img.shape[:2]
            roi_h = int(H * roi_ratio)
            roi_w = int(W * roi_w_ratio)

            cx = W // 2
            cy = center_y - SHIFT_Y

            y1 = max(0, (cy - roi_h // 2) )
            # y1 = max(0, Y1_MAX)
            y2 = min(H, (y1 + roi_h) )
            x1 = max(0, cx - roi_w // 2)
            x2 = min(W, x1 + roi_w)

            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            canvas[y1:y2, x1:x2] = roi

            cv2.imwrite(str(out_id_dir / img_path.name), canvas)
            
    # remove empty or too small folders


def write_variant_dataset(ratio):
    ## check if dataset already exists by reading fingerprint
      tag = f"ratio_{int(ratio * 100)}"
      for split in ["train", "val", "test"]:
          print(f"📂 Writing split: {split} with ROI ratio: {ratio}")
          write_split(
              src_root=SRC_DATASET_ROOT / split,
              center_y_json=CENTER_Y_JSON / f"center_of_eyes_points_{split}.json" ,
              dst_root=DST_DATASET_ROOT / tag / split,
              roi_ratio=ratio,
              roi_w_ratio=ROI_WIDTH_RATIO
          )
    

if __name__ == "__main__":  
  ## iterate over variants (ROIs)
    for ratio in ROI_RATIOS:
        tag = f"ratio_{int(ratio * 100)}"

        ## create partial face data variant
        print(f"\n📦 Creating ROI dataset: {tag}")
        write_variant_dataset(ratio)
        