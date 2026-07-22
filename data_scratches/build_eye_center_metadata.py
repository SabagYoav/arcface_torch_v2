"""
Compute eye-center-y metadata for the full glint360k dataset using buffalo_l
(insightface), for on-the-fly ROI cropping (no duplicated per-ratio image
folders — see training_multi_loops.py's on-the-fly dataset path).

Images are already 112x112 ArcFace-aligned. For each image we detect faces
(detection-only buffalo_l submodule — cheap, skips the 4 unused auxiliary
models), pick the most-central/largest face, and record:
    center_y = (left_eye_y + right_eye_y) / 2      (kps[0]=L eye, kps[1]=R eye)
Falls back to the canonical ArcFace template eye-y (51.69) with detected=False
when no face is found (rare — ~100% detection rate on a random sample).

Metadata is keyed by "<id>/<filename>" (NOT bare filename — glint360k id
folders all restart numbering at 0001.jpg, so bare-filename keys collide).

Parallelized 1 process per GPU (measured empirically as the throughput
sweet spot in this environment — CPU multiprocessing and >1 process/GPU both
scale worse due to a shared CPU-side preprocessing bottleneck). Resumable:
each shard writes progress incrementally and skips identities already marked
done on restart.

Usage:
    python3 data_scratches/build_eye_center_metadata.py [--limit N]
    python3 data_scratches/build_eye_center_metadata.py --merge
"""
import argparse
import json
import os
import time
from multiprocessing import Process
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/DATA/glint360k_download/glint360k_images")
OUT_DIR = Path("/DATA/glint360k_download/eye_center_metadata")
NUM_SHARDS = 4  # one process per GPU
DET_SIZE = (160, 160)
DET_THRESH = 0.5
TEMPLATE_EYE_Y = 51.69  # ArcFace 112 template canonical eye row
IMG_EXTS = {".jpg", ".jpeg", ".png"}
FLUSH_EVERY = 200  # identities


def shard_paths(shard_id):
    jsonl_path = OUT_DIR / f"shard_{shard_id}.jsonl"
    done_path = OUT_DIR / f"shard_{shard_id}.done_ids.txt"
    return jsonl_path, done_path


def eye_center_y(app, bgr):
    faces = app.get(bgr)
    if not faces:
        return TEMPLATE_EYE_Y, False
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


def run_shard(shard_id, all_ids, limit=None):
    # NOTE: LD_LIBRARY_PATH (for libcudnn.so.9, needed by onnxruntime's CUDA EP)
    # must be set in the shell environment *before* the python3 process starts —
    # setting os.environ here is too late, the dynamic linker has already
    # resolved its search path. See the wrapper shell invocation.
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=shard_id, det_size=DET_SIZE, det_thresh=DET_THRESH)

    jsonl_path, done_path = shard_paths(shard_id)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if done_path.exists():
        done_ids = set(done_path.read_text().splitlines())

    my_ids = [i for i in all_ids if int(i) % NUM_SHARDS == shard_id]
    if limit:
        my_ids = my_ids[:limit]
    remaining = [i for i in my_ids if i not in done_ids]
    print(f"[shard {shard_id}] {len(my_ids)} ids total, {len(remaining)} remaining", flush=True)

    n_done_ids = 0
    n_imgs = 0
    n_detected = 0
    t0 = time.time()
    with open(jsonl_path, "a") as jf, open(done_path, "a") as df:
        for id_name in remaining:
            id_dir = ROOT / id_name
            try:
                files = sorted(
                    f for f in os.listdir(id_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS
                )
            except FileNotFoundError:
                continue

            for fname in files:
                p = id_dir / fname
                bgr = cv2.imread(str(p))
                if bgr is None:
                    continue
                cy, detected = eye_center_y(app, bgr)
                rec = {"k": f"{id_name}/{fname}", "center_y": cy, "detected": detected}
                jf.write(json.dumps(rec) + "\n")
                n_imgs += 1
                n_detected += int(detected)

            df.write(id_name + "\n")
            n_done_ids += 1

            if n_done_ids % FLUSH_EVERY == 0:
                jf.flush()
                df.flush()
                dt = time.time() - t0
                print(
                    f"[shard {shard_id}] {n_done_ids}/{len(remaining)} ids, "
                    f"{n_imgs} imgs ({n_imgs/dt:.1f} img/s), "
                    f"detected {n_detected}/{n_imgs} ({100*n_detected/max(n_imgs,1):.1f}%)",
                    flush=True,
                )

    print(f"[shard {shard_id}] DONE: {n_done_ids} ids, {n_imgs} images", flush=True)


def merge():
    out_path = OUT_DIR / "center_eyes_metadata.json"
    merged = {}
    for shard_id in range(NUM_SHARDS):
        jsonl_path, _ = shard_paths(shard_id)
        if not jsonl_path.exists():
            continue
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                merged[rec["k"]] = {"detected": rec["detected"], "center_y": rec["center_y"]}
    with open(out_path, "w") as f:
        json.dump(merged, f)
    print(f"Merged {len(merged)} entries -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="limit ids per shard (testing)")
    ap.add_argument("--merge", action="store_true", help="merge shard JSONL files into one JSON")
    args = ap.parse_args()

    if args.merge:
        merge()
        return

    all_ids = sorted(os.listdir(ROOT))
    print(f"Total identities: {len(all_ids)}")

    procs = []
    for shard_id in range(NUM_SHARDS):
        p = Process(target=run_shard, args=(shard_id, all_ids, args.limit))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    merge()


if __name__ == "__main__":
    main()
