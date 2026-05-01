import os
import cv2
import json
import numpy as np
from PIL import Image
import face_alignment
from tqdm import tqdm
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')


def get_landmarks(img):
    # img = np.array(img_pil)
    lms = fa.get_landmarks(img)
    if lms is None:
        return None
    return lms[0]   # shape: (68, 2)

import json
def overite_file(fpath: str, data: dict):
    # Overwrite JSON file with provided data
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=4)

def load_metadata(fpath: str) -> dict:
    # Load metadata from JSON file, return empty dict if file doesn't exist
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    return data

def create_lms_metadata(root_dir):
    # lms_metadata = load_metadata(save_json_path)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    # Count total directories for progress bar
    print("Scanning directories to count total...")
    dir_list = [d for d, _, _ in os.walk(root_dir)]
    print(f"Start process with total directories: {len(dir_list)}")

    for dirpath, _, filenames in tqdm(os.walk(root_dir), total=len(dir_list), desc="Processing directories"):
        dir_lms_metadata = {}
        json_save_path = os.path.join('/DATA/faces/metadatas/glint_metadata', f"{os.path.basename(dirpath)}_lms.json")

        ## skip if directory is already processed   
        if os.path.exists(json_save_path):
            print(f"Skipping {dirpath}, {json_save_path} already exists.")
            continue

        ## process images in the directory
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in exts:
                image_path = os.path.join(dirpath, filename)
                lms = get_landmarks(cv2.imread(image_path))
                if lms is None:
                    print(f"Warning: No landmarks detected for image {image_path}")
                    continue
                dir_lms_metadata[image_path] = lms.tolist()

        # Save directory-specific landmarks metadata
        overite_file(os.path.join("/DATA/faces/metadatas/glint_metadata", f"{os.path.basename(dirpath)}_lms.json"), dir_lms_metadata)
    
    return dir_lms_metadata


if __name__ == "__main__":
    data_root = "/DATA/faces/glint360k_224"
    metadata = create_lms_metadata(data_root)