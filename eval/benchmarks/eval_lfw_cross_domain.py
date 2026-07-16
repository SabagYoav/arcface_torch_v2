"""
LFW verification for the ratio_15 ViT-S partial student vs R50 full-face teacher.

Rows in the comparison table
----------------------------
  teacher_full_vs_full     : cos(R50(fullA),  R50(fullB))            -> standard LFW upper bound
  cross_partial_vs_full    : sym cos(ViT-S(partialX), R50(fullY))    -> the cross-domain model under test
  student_partial_vs_partial: cos(ViT-S(partialA), ViT-S(partialB))  -> partial-only (same-domain student)

Metrics (per row)
-----------------
  acc_10fold        : official LFW protocol, per-fold best threshold, mean +/- std
  auc               : ROC AUC over all 6000 pairs
  best_acc / best_thr: single global best-threshold accuracy (matches ClipVerification)
  tar@far=1e-2/1e-3 : TAR at fixed FAR operating points
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

_this = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this, "..", ".."))
from backbones import get_model

ap = argparse.ArgumentParser()
ap.add_argument("--student", default="work_dirs/exp_clip_r50_vs_vit/clip_ratio_15/best_model.pt")
ap.add_argument("--student-net", default="vit_s_dp005_mask_0")
ap.add_argument("--teacher", default="work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt")
ap.add_argument("--teacher-net", default="r50")
ap.add_argument("--partial", default="partial_15", help="partial-set subdir name under the LFW root")
ap.add_argument("--tag", default=None, help="output filename tag (defaults to partial dir name)")
args = ap.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/media/yoav/Yoav/datasets/benchmarks/lfw")
FULL_DIR = ROOT / "full"
PARTIAL_DIR = ROOT / args.partial
PAIRS_TXT = ROOT / "pairs.txt"
TAG = args.tag or args.partial
OUT_DIR = Path("work_dirs/benchmarks/lfw"); OUT_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_CKPT = args.student
STUDENT_NET = args.student_net
TEACHER_CKPT = args.teacher
TEACHER_NET = args.teacher_net
EMB = 512
BATCH = 128

tfm = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_model(net, ckpt):
    m = get_model(net, dropout=0.0, fp16=True, num_features=EMB).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=True)
    m.eval()
    return m


def img_path(root, name, idx):
    return root / name / f"{name}_{int(idx):04d}.jpg"


def parse_pairs():
    """Return (pairs, folds). pairs[i]=(nameA,idxA,nameB,idxB,label), fold=i//600."""
    lines = PAIRS_TXT.read_text().strip().splitlines()
    hdr = lines[0].split()
    n_folds, per = int(hdr[0]), int(hdr[1])
    pairs, folds = [], []
    body = lines[1:]
    for i, ln in enumerate(body):
        t = ln.split()
        if len(t) == 3:
            pairs.append((t[0], t[1], t[0], t[2], 1))
        elif len(t) == 4:
            pairs.append((t[0], t[1], t[2], t[3], 0))
        else:
            continue
        folds.append(i // (2 * per))
    return pairs, np.array(folds), n_folds


@torch.no_grad()
def embed_all(model, root, keys):
    """keys: set of 'Name/idx'. Returns dict key->normalized emb (cpu numpy)."""
    keys = sorted(keys)
    embs = {}
    buf_imgs, buf_keys = [], []

    def flush():
        if not buf_imgs:
            return
        x = torch.stack(buf_imgs).to(DEVICE)
        with torch.no_grad():
            e = model(x)
        e = F.normalize(e, dim=1).cpu().numpy()
        for k, v in zip(buf_keys, e):
            embs[k] = v
        buf_imgs.clear(); buf_keys.clear()

    n_missing = 0
    for k in tqdm(keys, desc=f"embed {root.name}"):
        name, idx = k.rsplit("/", 1)
        p = img_path(root, name, idx)
        if not p.exists():
            n_missing += 1
            continue
        buf_imgs.append(tfm(Image.open(p).convert("RGB")))
        buf_keys.append(k)
        if len(buf_imgs) >= BATCH:
            flush()
    flush()
    return embs, n_missing


def metrics(scores, labels, folds, n_folds):
    scores = np.asarray(scores); labels = np.asarray(labels)
    # 10-fold: threshold chosen on the other 9 folds
    thr_grid = np.linspace(-1, 1, 4001)
    accs = []
    for f in range(n_folds):
        tr = folds != f; te = folds == f
        acc_tr = [( (scores[tr] >= t) == labels[tr] ).mean() for t in thr_grid]
        best_t = thr_grid[int(np.argmax(acc_tr))]
        accs.append((( (scores[te] >= best_t) == labels[te]).mean()))
    accs = np.array(accs)
    # global best threshold (ClipVerification style)
    acc_all = np.array([(((scores >= t) == labels).mean()) for t in thr_grid])
    gi = int(np.argmax(acc_all))
    # roc / auc / tar@far
    fpr, tpr, thr = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    def tar_at(far):
        idx = np.where(fpr <= far)[0]
        return float(tpr[idx[-1]]) if len(idx) else 0.0
    return {
        "acc_10fold_mean": float(accs.mean()),
        "acc_10fold_std": float(accs.std()),
        "best_acc": float(acc_all[gi]),
        "best_thr": float(thr_grid[gi]),
        "auc": float(roc_auc),
        "tar@far=1e-2": tar_at(1e-2),
        "tar@far=1e-3": tar_at(1e-3),
        "_roc": (fpr.tolist(), tpr.tolist()),
    }


def main():
    print(f"Student: {STUDENT_CKPT} ({STUDENT_NET})")
    print(f"Teacher: {TEACHER_CKPT} ({TEACHER_NET})")
    print(f"Partial set: {PARTIAL_DIR.name}")
    pairs, folds, n_folds = parse_pairs()
    print(f"Parsed {len(pairs)} pairs in {n_folds} folds")

    keys = set()
    for a, ia, b, ib, _ in pairs:
        keys.add(f"{a}/{ia}"); keys.add(f"{b}/{ib}")

    print("Loading models ...")
    student = load_model(STUDENT_NET, STUDENT_CKPT)
    teacher = load_model(TEACHER_NET, TEACHER_CKPT)

    full_t, miss_ft = embed_all(teacher, FULL_DIR, keys)      # R50 on full
    part_s, miss_ps = embed_all(student, PARTIAL_DIR, keys)   # ViT-S on partial

    rows = {}
    dropped = 0
    S_tt, S_cross, S_pp, y, fold_ok = [], [], [], [], []
    for (a, ia, b, ib, lbl), fd in zip(pairs, folds):
        ka, kb = f"{a}/{ia}", f"{b}/{ib}"
        if ka not in full_t or kb not in full_t or ka not in part_s or kb not in part_s:
            dropped += 1
            continue
        fa, fb = full_t[ka], full_t[kb]
        pa, pb = part_s[ka], part_s[kb]
        S_tt.append(float(fa @ fb))
        S_cross.append(0.5 * (float(pa @ fb) + float(pb @ fa)))
        S_pp.append(float(pa @ pb))
        y.append(lbl); fold_ok.append(fd)
    y = np.array(y); fold_ok = np.array(fold_ok)
    print(f"Usable pairs: {len(y)}  dropped(missing img): {dropped}")

    rows["teacher_full_vs_full"] = metrics(S_tt, y, fold_ok, n_folds)
    rows["cross_partial_vs_full"] = metrics(S_cross, y, fold_ok, n_folds)
    rows["student_partial_vs_partial"] = metrics(S_pp, y, fold_ok, n_folds)

    # ---- table ----
    hdr = f"{'row':<28}{'acc(10fold)':<18}{'AUC':<9}{'best_acc':<10}{'TAR@1e-2':<10}{'TAR@1e-3':<10}"
    print("\n" + "=" * len(hdr)); print(f"LFW  (partial = {PARTIAL_DIR.name}, cross-domain)"); print("=" * len(hdr))
    print(hdr); print("-" * len(hdr))
    for k, m in rows.items():
        print(f"{k:<28}{m['acc_10fold_mean']*100:6.2f} +/- {m['acc_10fold_std']*100:4.2f}   "
              f"{m['auc']:<9.4f}{m['best_acc']*100:<10.2f}{m['tar@far=1e-2']*100:<10.2f}{m['tar@far=1e-3']*100:<10.2f}")

    save = {k: {kk: vv for kk, vv in m.items() if kk != "_roc"} for k, m in rows.items()}
    save["_meta"] = {"n_pairs": int(len(y)), "dropped": int(dropped),
                     "missing_full": int(miss_ft), "missing_partial": int(miss_ps),
                     "student_ckpt": STUDENT_CKPT, "teacher_ckpt": TEACHER_CKPT}
    res_path = OUT_DIR / f"lfw_results_{TAG}.json"
    with open(res_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved -> {res_path}")

    # ---- ROC plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 7))
    for k, m in rows.items():
        fpr, tpr = m["_roc"]
        plt.plot(fpr, tpr, label=f"{k} (AUC={m['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xscale("log"); plt.xlim(1e-4, 1); plt.ylim(0, 1.01)
    plt.xlabel("FAR (log)"); plt.ylabel("TAR")
    plt.title(f"LFW ROC — partial({PARTIAL_DIR.name} ViT-S) vs full(R50)")
    plt.legend(loc="lower right"); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    roc_path = OUT_DIR / f"lfw_roc_{TAG}.png"
    plt.savefig(roc_path, dpi=180); plt.close()
    print(f"Saved -> {roc_path}")


if __name__ == "__main__":
    main()
