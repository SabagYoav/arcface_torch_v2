"""
CFP preprocessing (mirrors the LFW pipeline).

- Align every referenced frontal/profile image to the ArcFace 112 template
  (insightface 5-pt landmarks + norm_crop)                -> full/<key>.jpg   (teacher domain)
- Build the clean literal-20% eye-band partial                -> partial_20/<key>.jpg (student domain)
- Emit official 10-split pair manifests for CFP-FF and CFP-FP -> pairs_FF.json / pairs_FP.json
  as lists of [keyA, keyB, label, fold].

key = "F{idx}" (frontal, Pair_list_F.txt) or "P{idx}" (profile, Pair_list_P.txt).
"""
import json
from pathlib import Path
import numpy as np, cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils import face_align

ROOT = Path("/media/yoav/Yoav/datasets/benchmarks/cfp")
RAW = ROOT / "raw" / "cfp-dataset"
IMG = RAW / "Data" / "Images"
PROTO = RAW / "Protocol"
FULL = ROOT / "full"
PART = ROOT / "partial_20"
RATIO = 0.20

app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(224, 224))


def best_face(faces, shape):
    H, W = shape[:2]; c = np.array([W / 2, H / 2])
    def score(f):
        bb = f.bbox
        area = (bb[2]-bb[0])*(bb[3]-bb[1])
        cen = np.hypot((bb[0]+bb[2])/2-c[0], (bb[1]+bb[3])/2-c[1])
        return area - 2.0*cen
    return max(faces, key=score)


def parse_pair_list(fn):
    d = {}
    for ln in (PROTO / fn).read_text().strip().splitlines():
        i, rel = ln.split()
        d[int(i)] = (IMG / rel.replace("../Data/Images/", "")).resolve()
    return d


def align_and_partial(src_path, key):
    bgr = cv2.imread(str(src_path))
    if bgr is None:
        return False
    faces = app.get(bgr)
    if not faces:
        return False
    f = best_face(faces, bgr.shape)
    aligned = face_align.norm_crop(bgr, landmark=f.kps, image_size=112)  # 112x112 arcface
    # eye center on the aligned crop (re-detect; fallback to template 51.69)
    af = app.get(aligned)
    if af:
        k = best_face(af, aligned.shape).kps
        py = float((k[0][1] + k[1][1]) / 2.0)
    else:
        py = 51.69
    # clean literal-20% band, centered on eyes, full width
    H, W = aligned.shape[:2]
    roi_h = int(round(RATIO * H)); cy = int(round(py))
    y1 = max(0, cy - roi_h // 2); y2 = min(H, y1 + roi_h)
    part = np.zeros((H, W, 3), dtype=np.uint8); part[y1:y2, :] = aligned[y1:y2, :]

    (FULL / f"{key}.jpg").parent.mkdir(parents=True, exist_ok=True)
    (PART / f"{key}.jpg").parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FULL / f"{key}.jpg"), aligned)
    cv2.imwrite(str(PART / f"{key}.jpg"), part)
    return True


def main():
    F = parse_pair_list("Pair_list_F.txt")
    P = parse_pair_list("Pair_list_P.txt")
    print(f"frontal={len(F)} profile={len(P)}")

    ok, fail = 0, []
    for idx, p in tqdm(sorted(F.items()), desc="align frontal"):
        (align_and_partial(p, f"F{idx}") and (ok := ok + 1)) or fail.append(f"F{idx}")
    for idx, p in tqdm(sorted(P.items()), desc="align profile"):
        (align_and_partial(p, f"P{idx}") and (ok := ok + 1)) or fail.append(f"P{idx}")
    print(f"aligned ok={ok}  failed={len(fail)}")

    def build(kind, keyfmt):
        pairs = []
        for split in range(1, 11):
            d = PROTO / "Split" / kind / f"{split:02d}"
            for fn, lbl in [("same.txt", 1), ("diff.txt", 0)]:
                for ln in (d / fn).read_text().strip().splitlines():
                    a, b = ln.split(",")
                    pairs.append([keyfmt[0] + a, keyfmt[1] + b, lbl, split - 1])
        json.dump(pairs, open(ROOT / f"pairs_{kind}.json", "w"))
        n_pos = sum(1 for x in pairs if x[2] == 1)
        print(f"{kind}: {len(pairs)} pairs ({n_pos} pos) -> pairs_{kind}.json")

    build("FF", ("F", "F"))   # frontal-frontal
    build("FP", ("F", "P"))   # frontal-profile

    if fail:
        json.dump(fail, open(ROOT / "align_failures.json", "w"))
        print(f"failures logged -> {ROOT/'align_failures.json'}")


if __name__ == "__main__":
    main()
