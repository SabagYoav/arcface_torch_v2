"""Standard cross-modal verification benchmark (STANDALONE — does not import or
modify any training-pipeline function).

Protocol
--------
For each ROI ratio with a trained student checkpoint:
  * probe     = partial-face crop  -> STUDENT model (that ratio's best_model.pt)
  * reference = full face          -> TEACHER model (R50)
Both embeddings are L2-normalized; the pair score is their cosine similarity.

Pairs (balanced):
  * positive = same identity, DIFFERENT photo (partial of photo A vs full of photo B,
               A != B) — the trivial same-photo cross-modal pair is excluded.
  * negative = different identities.

Metrics (LFW-style, no threshold leakage):
  * Accuracy  : 10-fold CV — threshold chosen on the 9 train folds, applied to the
                held-out fold; reported as mean ± std.
  * TAR@FAR   : true-accept rate at FAR = 1e-1, 1e-2, 1e-3 (from the global ROC).
  * AUC       : area under the ROC curve.

Only reads model weights and images; builds its own deterministic (no-aug) transform.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backbones import get_model  # read-only reuse; not modified

# ------------------------- config -------------------------
EXP_DIR       = "work_dirs/exp_clip_r50_vs_vit"          # <- use this experiment only
STUDENT_NET   = "vit_s_dp005_mask_0"
TEACHER_NET   = "r50"
TEACHER_CKPT  = "work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt"
EMB_SIZE      = 512
RATIOS        = [1.0, 0.3, 0.2, 0.15]                    # ratios present in EXP_DIR
FULLFACE_VAL  = "/media/yoav/Yoav/datasets/glint360k/fullface_subset/val"
PARTIAL_VAL   = "/media/yoav/Yoav/datasets/variants_dataset_subset/ratio_{r}/val"
N_POS = N_NEG = 3000
N_FOLDS       = 10
FAR_TARGETS   = [1e-1, 1e-2, 1e-3]
SEED          = 2048
BATCH         = 128
DEVICE        = "cuda"
IMG_EXTS      = {".jpg", ".jpeg", ".png"}

_tf = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


@torch.no_grad()
def embed_split(root, model):
    """Return (emb[N,512] normalized, labels[N] int, fnames[N] str) for an ImageFolder-like dir."""
    root = Path(root)
    paths, labels, fnames = [], [], []
    for id_dir in sorted(root.iterdir()):
        if not id_dir.is_dir() or not id_dir.name.isdigit():
            continue
        for img in sorted(id_dir.iterdir()):
            if img.suffix.lower() in IMG_EXTS:
                paths.append(img)
                labels.append(int(id_dir.name))
                fnames.append(img.name)

    embs = []
    buf = []
    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(DEVICE)
        e = F.normalize(model(x).float(), dim=1).cpu()
        embs.append(e)
        buf.clear()
    for p in paths:
        buf.append(_tf(Image.open(p).convert("RGB")))
        if len(buf) >= BATCH:
            flush()
    flush()
    return torch.cat(embs).numpy(), np.array(labels), np.array(fnames)


def build_pairs(p_lab, p_fn, f_lab, f_fn, rng):
    """Cross-modal balanced pairs. Returns (idx_partial, idx_full, y)."""
    # id -> indices
    p_by_id, f_by_id = {}, {}
    for i, l in enumerate(p_lab):
        p_by_id.setdefault(l, []).append(i)
    for j, l in enumerate(f_lab):
        f_by_id.setdefault(l, []).append(j)
    common = [i for i in p_by_id if i in f_by_id]

    pi, fj, y = [], [], []
    # positives: same id, different photo (partial fname != full fname)
    tries = 0
    while sum(y) < N_POS and tries < N_POS * 50:
        tries += 1
        cid = common[rng.integers(len(common))]
        a = p_by_id[cid][rng.integers(len(p_by_id[cid]))]
        b = f_by_id[cid][rng.integers(len(f_by_id[cid]))]
        if p_fn[a] == f_fn[b]:            # skip the trivial same-photo cross-modal pair
            continue
        pi.append(a); fj.append(b); y.append(1)
    # negatives: different identities
    ids = list(f_by_id.keys())
    n_neg = 0
    while n_neg < N_NEG:
        cid = common[rng.integers(len(common))]
        did = ids[rng.integers(len(ids))]
        if did == cid:
            continue
        a = p_by_id[cid][rng.integers(len(p_by_id[cid]))]
        b = f_by_id[did][rng.integers(len(f_by_id[did]))]
        pi.append(a); fj.append(b); y.append(0)
        n_neg += 1
    return np.array(pi), np.array(fj), np.array(y)


def kfold_accuracy(scores, y, rng):
    """LFW-style: per fold, pick threshold maximizing accuracy on the other folds."""
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, N_FOLDS)
    accs, thrs = [], []
    cand = np.unique(scores)
    for k in range(N_FOLDS):
        test = folds[k]
        train = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
        # best threshold on train
        best_t, best_a = 0.0, -1
        for t in cand:
            a = ((scores[train] >= t) == y[train]).mean()
            if a > best_a:
                best_a, best_t = a, t
        accs.append(((scores[test] >= best_t) == y[test]).mean())
        thrs.append(best_t)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(thrs))


def tar_at_far(scores, y):
    fpr, tpr, _ = roc_curve(y, scores)
    out = {}
    for far in FAR_TARGETS:
        out[far] = float(np.interp(far, fpr, tpr))
    return out


def main():
    torch.backends.cudnn.benchmark = True
    teacher = get_model(TEACHER_NET, dropout=0.0, fp16=False, num_features=EMB_SIZE).to(DEVICE).eval()
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location="cpu"))
    print("Teacher (R50) loaded. Embedding full-face val once...")
    f_emb, f_lab, f_fn = embed_split(FULLFACE_VAL, teacher)
    print(f"  full-face val: {len(f_lab)} images, {len(set(f_lab))} ids")

    rows = []
    for r in RATIOS:
        tag = f"ratio_{int(round(r * 100))}"
        ckpt = Path(EXP_DIR) / f"clip_{tag}" / "best_model.pt"
        if not ckpt.exists():
            print(f"[skip] {tag}: no checkpoint at {ckpt}")
            continue
        student = get_model(STUDENT_NET, dropout=0.0, fp16=False, num_features=EMB_SIZE).to(DEVICE).eval()
        student.load_state_dict(torch.load(ckpt, map_location="cpu"))
        p_emb, p_lab, p_fn = embed_split(PARTIAL_VAL.format(r=int(round(r * 100))), student)

        rng = np.random.default_rng(SEED)
        pi, fj, y = build_pairs(p_lab, p_fn, f_lab, f_fn, rng)
        scores = np.sum(p_emb[pi] * f_emb[fj], axis=1)   # cosine (already normalized)

        acc, std, thr = kfold_accuracy(scores, y, np.random.default_rng(SEED))
        auc = roc_auc_score(y, scores)
        tar = tar_at_far(scores, y)
        rows.append((tag, int(round(r * 100)), len(y), acc, std, tar[1e-1], tar[1e-2], tar[1e-3], auc, thr))
        print(f"  {tag}: acc={acc:.4f}±{std:.4f} AUC={auc:.4f} "
              f"TAR@FAR[1e-1,1e-2,1e-3]=[{tar[1e-1]:.3f},{tar[1e-2]:.3f},{tar[1e-3]:.3f}] thr={thr:.3f}")

    # ---- table ----
    print("\n=== Standard cross-modal verification (ViT student partial vs R50 teacher full) ===")
    hdr = f"{'ROI':>5} | {'#pairs':>6} | {'Acc±std':>15} | {'TAR@1e-1':>8} | {'TAR@1e-2':>8} | {'TAR@1e-3':>8} | {'AUC':>6} | {'thr':>6}"
    print(hdr); print("-" * len(hdr))
    for tag, roi, npairs, acc, std, t1, t2, t3, auc, thr in rows:
        print(f"{roi:>4}% | {npairs:>6} | {acc*100:>6.2f} ± {std*100:>4.2f}% | "
              f"{t1*100:>7.2f}% | {t2*100:>7.2f}% | {t3*100:>7.2f}% | {auc:>6.4f} | {thr:>6.3f}")


if __name__ == "__main__":
    main()
