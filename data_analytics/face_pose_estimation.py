from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os
import pandas as pd

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_thresh=0.5, det_size=(224, 224))

def analyze(path):
    img = cv2.imread(path)
    if img is None:
        return None

    faces = app.get(img)
    if not faces:
        return None

    face = faces[0]
    yaw, pitch, roll = face.pose  # FIXED

    return {
        "path": path,
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
    }

def iterate_image_dataset(root_dir, exts={".jpg", ".png", ".jpeg"}):
    rows = []
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            _, ext = os.path.splitext(f)
            if ext.lower() in exts:
                path = os.path.join(subdir, f)
                pose = analyze(path)
                if pose is not None:
                    rows.append(pose)     # append dict, NOT tuple
                else:
                    print(f"Could not analyze image: {path}")
            print(f"Processed {len(rows)} images so far...")
    return rows

rows = iterate_image_dataset("/DATA_FP/glint360k/imageFolder_split_fullface/train")
df = pd.DataFrame(rows)

df.to_json("glint_train_set_pose_results.json", orient="records", indent=2)

print("Done")
