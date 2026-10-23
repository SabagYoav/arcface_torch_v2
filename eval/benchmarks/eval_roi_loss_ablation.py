"""InfoNCE vs MSE (swin_base students, 36k-id / 10-per-id glint subset, ROI 0.15)
on harder benchmarks + open-set identification.

Verification (LFW, CPLFW, CFP-FP), cross-modal, same protocol as training-time
ClipVerification: probe = ROI-0.15 crop -> student, reference = full face ->
R50 teacher, symmetric pair score 0.5*(cos(pA,fB) + cos(pB,fA)), 10-fold mean
accuracy on the benchmark's official folds (same threshold grid), AUC, TAR@FAR.
ROI crops use the same buffalo_l eye-center detection + crop_roi as training.
LFW is included as a sanity check against the logged final-epoch numbers.

Open-set identification on the HELD-OUT glint val identities (9k ids, unseen
by every model here): ids split 50/50 (seeded) into enrolled / never-enrolled.
Gallery = 1 full-face image per enrolled id (teacher). Probes = up to 5
ROI-0.15 crops per id (student) from both halves. The never-enrolled
identities are split again 50/50 BY IDENTITY into a calibration half and a
test half: the threshold is the score at which 0.1% of the calibration
half's probes are accepted, and FAR (on the test half's never-enrolled
probes) and DIR (fraction of enrolled probes whose rank-1 match is correct
AND above the threshold) are reported with that threshold. Also reported
with the halves swapped.

Rows: each student, the zero-shot R50 teacher (partial AND full through R50),
and teacher full-vs-full as an upper anchor.

Usage (in the devcontainer):
  python3 eval/benchmarks/eval_roi_loss_ablation.py
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backbones import get_model
from dataset import crop_roi, get_transform
from eval.benchmarks.lfw_bin import TEMPLATE_EYE_Y, compute_or_load_eye_centers, decode_lfw_bin, pair_folds
from utils.clip_verifications_utils import ClipVerification

RATIO = 0.15
TEACHER = ("r50", "16backbone.pth")
STUDENTS = {
    "InfoNCE (swin_base)": ("swin_base", "work_dirs/exp_arch_comparison_10perid/swin_base_ratio_15/model.pt"),
    "MSE (swin_base)": ("swin_base", "work_dirs/exp_loss_ablation/swin_base_mse_ratio_15/model.pt"),
}
BENCH = Path("/DATA/face_ReId_benchmarks")
GLINT_VAL = Path("/DATA/glint360k_arch_comparison/val")
GLINT_VAL_META = Path("/DATA/glint360k_arch_comparison/val_eye_center_metadata.json")
OUT = Path("work_dirs/benchmarks/roi_loss_ablation")
FAR_TARGET = 1e-3
PROBES_PER_ID = 5
SEED = 42
THRESHOLDS = ClipVerification.THRESHOLDS
transform = get_transform(augmentations=False)


# ---------------------------------------------------------------- data
def load_cplfw():
    root = BENCH / "val"
    labels, paths = [], []
    for line in open(root / "cplfw_ann.txt"):
        lab, a, b = line.split()
        labels.append(lab == "1")
        paths += [root / a, root / b]
    images = [cv2.imread(str(p)) for p in paths]
    return images, np.array(labels), pair_folds(len(labels))


def load_cfp_fp():
    root = BENCH / "cfp"
    pairs = json.load(open(root / "pairs_FP.json"))
    images = []
    for a, b, _, _ in pairs:
        images += [cv2.imread(str(root / "full" / f"{a}.jpg")), cv2.imread(str(root / "full" / f"{b}.jpg"))]
    return images, np.array([p[2] == 1 for p in pairs]), np.array([p[3] for p in pairs])


def load_lfw():
    images, issame = decode_lfw_bin()
    return images, np.array(issame, dtype=bool), pair_folds(len(issame))


def eye_centers(name, images, app):
    """Same detection + face selection as lfw_bin.compute_or_load_eye_centers, cached per benchmark."""
    cache = OUT / f"eye_centers_{name}.json"
    if cache.exists():
        return json.load(open(cache))
    H, W = images[0].shape[:2]
    c = np.array([W / 2, H / 2])
    out = []
    for bgr in images:
        faces = app.get(bgr)
        if not faces:
            out.append(TEMPLATE_EYE_Y)
            continue
        f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                - 2.0 * np.hypot((f.bbox[0] + f.bbox[2]) / 2 - c[0], (f.bbox[1] + f.bbox[3]) / 2 - c[1]))
        out.append(float((f.kps[0][1] + f.kps[1][1]) / 2.0))
    json.dump(out, open(cache, "w"))
    return out


def to_tensors(images, centers=None):
    out = []
    for i, bgr in enumerate(images):
        if centers is not None:
            bgr = crop_roi(bgr, RATIO, centers[i])
        out.append(transform(ClipVerification._to_pil(bgr)))
    return torch.stack(out)


# ---------------------------------------------------------------- models
def load_model(net, ckpt):
    m = get_model(net, dropout=0.0, fp16=True, num_features=512).cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda"))
    return m.eval()


@torch.no_grad()
def embed(model, tensor, bs=256):
    return torch.cat([F.normalize(model(tensor[i:i + bs].cuda())).float().cpu()
                      for i in range(0, len(tensor), bs)])


# ---------------------------------------------------------------- metrics
def verification_metrics(scores, labels, folds):
    accs = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        t = THRESHOLDS[int(np.argmax([((scores[tr] >= t) == labels[tr]).mean() for t in THRESHOLDS]))]
        accs.append(((scores[te] >= t) == labels[te]).mean())
    fpr, tpr, _ = roc_curve(labels, scores)
    tar = lambda far: float(tpr[np.where(fpr <= far)[0][-1]])
    return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "auc": float(auc(fpr, tpr)), "tar@1e-2": tar(1e-2), "tar@1e-3": tar(1e-3)}


def cross_scores(Ep, Ef):
    a, b = np.arange(0, len(Ef), 2), np.arange(1, len(Ef), 2)
    return 0.5 * ((Ep[a] * Ef[b]).sum(1) + (Ep[b] * Ef[a]).sum(1)).numpy()


def dir_at_far(G, g_labels, P, p_labels, p_mated, p_calib, far=FAR_TARGET):
    """p_calib marks the never-enrolled probes used to pick the threshold;
    the remaining never-enrolled probes are the held-out test half."""
    S = (P @ G.T).numpy()
    top = S.argmax(1)
    top_score = S[np.arange(len(S)), top]
    correct = g_labels[top] == p_labels
    m = p_mated

    def at(calib, test):
        nm = np.sort(top_score[calib])[::-1]
        k = int(np.floor(far * len(nm)))   # allowed false accepts on the calibration half
        thr = nm[k]                        # accept iff score > thr  ->  <= k false accepts
        return {"dir": float((correct[m] & (top_score[m] > thr)).mean()),
                "far_test": float((top_score[test] > thr).mean()),
                "far_calib": float((top_score[calib] > thr).mean()),
                "thr": float(thr), "n_calib": int(calib.sum()), "n_test": int(test.sum())}

    a, b = ~m & p_calib, ~m & ~p_calib
    return {"rank1": float(correct[m].mean()), "n_mated": int(m.sum()),
            "A->B": at(a, b), "B->A": at(b, a)}


# ---------------------------------------------------------------- open-set split
def glint_open_set():
    meta = json.load(open(GLINT_VAL_META))
    by_id = {}
    for k, v in meta.items():
        if v["detected"]:
            by_id.setdefault(k.split("/", 1)[0], []).append(k)
    ids = sorted(by_id, key=int)
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    enrolled = set(ids[: len(ids) // 2])
    gallery, probes = [], []
    for i in ids:
        keys = sorted(by_id[i])
        rng.shuffle(keys)
        if i in enrolled:
            gallery.append(keys[0])
            keys = keys[1:]
        probes += [(k, i in enrolled) for k in keys[:PROBES_PER_ID]]
    # Calibration/test split of the never-enrolled identities (by identity, so no
    # person contributes probes to both halves). Separate RNG: gallery/probe
    # selection above is unchanged.
    unknown = sorted((i for i in ids if i not in enrolled), key=int)
    np.random.default_rng(SEED + 1).shuffle(unknown)
    calib_ids = set(unknown[: len(unknown) // 2])
    load = lambda ks: [cv2.imread(str(GLINT_VAL / k)) for k in ks]
    lab = lambda ks: np.array([int(k.split("/", 1)[0]) for k in ks])
    pk = [k for k, _ in probes]
    return {
        "gallery_full": to_tensors(load(gallery)),
        "g_labels": lab(gallery),
        "probe_partial": to_tensors(load(pk), [meta[k]["center_y"] for k in pk]),
        "probe_full": to_tensors(load(pk)),
        "p_labels": lab(pk),
        "p_mated": np.array([m for _, m in probes]),
        "p_calib": np.array([k.split("/", 1)[0] in calib_ids for k, _ in probes]),
        "n_enrolled": len(enrolled), "n_ids": len(ids),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app = ClipVerification._build_detector()

    benches = {}
    for name, loader in [("LFW", load_lfw), ("CPLFW", load_cplfw), ("CFP-FP", load_cfp_fp)]:
        images, labels, folds = loader()
        assert all(im is not None for im in images), f"{name}: unreadable image"
        if name == "LFW":  # the exact cache training-time ClipVerification uses
            centers = [c["center_y"] for c in compute_or_load_eye_centers(images, app)]
        else:
            centers = eye_centers(name, images, app)
        benches[name] = {"full": to_tensors(images), "partial": to_tensors(images, centers),
                         "labels": labels, "folds": folds}
        print(f"{name}: {len(labels)} pairs, {len(images)} images", flush=True)
    osd = glint_open_set()
    print(f"open-set: {osd['n_ids']} ids ({osd['n_enrolled']} enrolled), gallery {len(osd['g_labels'])}, "
          f"probes {len(osd['p_labels'])} ({osd['p_mated'].sum()} mated)", flush=True)

    teacher = load_model(*TEACHER)
    t_full = {n: embed(teacher, b["full"]) for n, b in benches.items()}
    G = embed(teacher, osd["gallery_full"])

    rows = {}
    def evaluate(row, model_partial, probe_full_by=None):
        r = {}
        for n, b in benches.items():
            Ep = embed(model_partial, b["partial"]) if probe_full_by is None else embed(probe_full_by, b["full"])
            r[n] = verification_metrics(cross_scores(Ep, t_full[n]), b["labels"], b["folds"])
        P = embed(model_partial, osd["probe_partial"]) if probe_full_by is None else embed(probe_full_by, osd["probe_full"])
        r["open_set"] = dir_at_far(G, osd["g_labels"], P, osd["p_labels"], osd["p_mated"], osd["p_calib"])
        rows[row] = r
        print(f"done: {row}", flush=True)

    for row, (net, ckpt) in STUDENTS.items():
        m = load_model(net, ckpt)
        evaluate(row, m)
        del m
        torch.cuda.empty_cache()
    evaluate("Zero-shot R50 (partial->R50)", teacher)
    evaluate("R50 full vs full (anchor)", teacher, probe_full_by=teacher)

    json.dump(rows, open(OUT / "results.json", "w"), indent=1)

    print(f"\nROI {RATIO} partial (probe) vs full (reference) -- 10-fold acc mean±std | TAR@1e-3")
    print(f"{'model':32s}" + "".join(f"{n:>24s}" for n in benches))
    for row, r in rows.items():
        print(f"{row:32s}" + "".join(
            f"{r[n]['acc_mean']:.4f}±{r[n]['acc_std']:.4f} | {r[n]['tar@1e-3']:.3f}".rjust(24) for n in benches))

    o = rows[next(iter(rows))]["open_set"]
    print(f"\nOpen-set (glint val): threshold at FAR {FAR_TARGET:.1%} on the calibration half of the "
          f"never-enrolled probes; FAR measured on the other half. {o['n_mated']} mated probes; "
          f"never-enrolled halves {o['A->B']['n_calib']} / {o['A->B']['n_test']} probes.")
    print(f"{'model':32s}{'rank-1':>8s}" + "".join(f"{d + ' DIR':>12s}{d + ' FAR':>12s}" for d in ("A->B", "B->A")))
    for row, r in rows.items():
        o = r["open_set"]
        print(f"{row:32s}{o['rank1']:>8.4f}" + "".join(
            f"{o[d]['dir']:>12.4f}{o[d]['far_test']:>12.5f}" for d in ("A->B", "B->A")))
    print(f"-> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
