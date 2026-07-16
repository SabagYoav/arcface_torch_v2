"""
Generic cross-domain verification over a pairs manifest.

Manifest JSON: list of [keyA, keyB, label, fold].
Image dirs are flat: <full_dir>/<key>.jpg (teacher domain) and <partial_dir>/<key>.jpg (student domain).

Rows: teacher_full_vs_full | cross_partial_vs_full (symmetric) | student_partial_vs_partial.
Metrics: N-fold acc (per-fold best threshold), AUC, global best_acc, TAR@FAR.

Usage:
  eval_pairs_cross_domain.py --root <bench_root> --pairs pairs_FP.json \
      --student <ckpt> --student-net vit_s_dp005_mask_0 \
      --teacher <ckpt> --partial partial_20 --tag CFP_FP
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

_this = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this, "..", ".."))
from backbones import get_model

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--pairs", required=True)
ap.add_argument("--student", required=True)
ap.add_argument("--student-net", default="vit_s_dp005_mask_0")
ap.add_argument("--teacher", default="work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt")
ap.add_argument("--teacher-net", default="r50")
ap.add_argument("--partial", default="partial_20")
ap.add_argument("--tag", required=True)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(args.root)
FULL = ROOT / "full"
PART = ROOT / args.partial
OUT = Path("work_dirs/benchmarks") / ROOT.name; OUT.mkdir(parents=True, exist_ok=True)
EMB, BATCH = 512, 128
tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)])


def load(net, ckpt):
    m = get_model(net, dropout=0.0, fp16=True, num_features=EMB).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True), strict=True)
    return m.eval()


@torch.no_grad()
def embed(model, root, keys):
    out = {}; bi, bk = [], []
    def flush():
        if not bi: return
        e = F.normalize(model(torch.stack(bi).to(DEV)), dim=1).cpu().numpy()
        for k, v in zip(bk, e): out[k] = v
        bi.clear(); bk.clear()
    miss = 0
    for k in tqdm(sorted(keys), desc=f"embed {root.name}"):
        p = root / f"{k}.jpg"
        if not p.exists(): miss += 1; continue
        bi.append(tfm(Image.open(p).convert("RGB"))); bk.append(k)
        if len(bi) >= BATCH: flush()
    flush()
    return out, miss


def metrics(scores, labels, folds):
    scores, labels, folds = map(np.asarray, (scores, labels, folds))
    nf = int(folds.max()) + 1
    grid = np.linspace(-1, 1, 4001)
    accs = []
    for f in range(nf):
        tr, te = folds != f, folds == f
        bt = grid[int(np.argmax([((scores[tr] >= t) == labels[tr]).mean() for t in grid]))]
        accs.append(((scores[te] >= bt) == labels[te]).mean())
    accs = np.array(accs)
    acc_all = np.array([((scores >= t) == labels).mean() for t in grid])
    gi = int(np.argmax(acc_all))
    fpr, tpr, _ = roc_curve(labels, scores); roc_auc = auc(fpr, tpr)
    tar = lambda far: float(tpr[np.where(fpr <= far)[0][-1]]) if np.any(fpr <= far) else 0.0
    return {"acc_nfold_mean": float(accs.mean()), "acc_nfold_std": float(accs.std()),
            "best_acc": float(acc_all[gi]), "best_thr": float(grid[gi]), "auc": float(roc_auc),
            "tar@far=1e-2": tar(1e-2), "tar@far=1e-3": tar(1e-3), "_roc": (fpr.tolist(), tpr.tolist())}


def main():
    pairs = json.load(open(ROOT / args.pairs))
    keys = set(k for p in pairs for k in p[:2])
    print(f"{args.tag}: {len(pairs)} pairs, {len(keys)} unique imgs")
    print(f"Student {args.student} ({args.student_net}) | partial={args.partial}")
    student, teacher = load(args.student_net, args.student), load(args.teacher_net, args.teacher)
    ft, mft = embed(teacher, FULL, keys)
    ps, mps = embed(student, PART, keys)

    Stt, Scr, Spp, y, fo = [], [], [], [], []
    drop = 0
    for a, b, lbl, fd in pairs:
        if a not in ft or b not in ft or a not in ps or b not in ps: drop += 1; continue
        fa, fb, pa, pb = ft[a], ft[b], ps[a], ps[b]
        Stt.append(float(fa @ fb))
        Scr.append(0.5*(float(pa @ fb) + float(pb @ fa)))
        Spp.append(float(pa @ pb)); y.append(lbl); fo.append(fd)
    print(f"usable={len(y)} dropped={drop}")
    rows = {"teacher_full_vs_full": metrics(Stt, y, fo),
            "cross_partial_vs_full": metrics(Scr, y, fo),
            "student_partial_vs_partial": metrics(Spp, y, fo)}

    hdr = f"{'row':<28}{'acc(nfold)':<18}{'AUC':<9}{'best_acc':<10}{'TAR@1e-2':<10}{'TAR@1e-3':<10}"
    print("\n" + "=" * len(hdr)); print(f"{args.tag}  (partial={args.partial}, cross-domain)"); print("=" * len(hdr))
    print(hdr); print("-" * len(hdr))
    for k, m in rows.items():
        print(f"{k:<28}{m['acc_nfold_mean']*100:6.2f} +/- {m['acc_nfold_std']*100:4.2f}   "
              f"{m['auc']:<9.4f}{m['best_acc']*100:<10.2f}{m['tar@far=1e-2']*100:<10.2f}{m['tar@far=1e-3']*100:<10.2f}")

    save = {k: {kk: vv for kk, vv in m.items() if kk != "_roc"} for k, m in rows.items()}
    save["_meta"] = {"n_pairs": len(y), "dropped": drop, "missing_full": mft, "missing_partial": mps,
                     "student": args.student, "partial": args.partial}
    json.dump(save, open(OUT / f"results_{args.tag}.json", "w"), indent=2)
    print(f"\nSaved -> {OUT/f'results_{args.tag}.json'}")


if __name__ == "__main__":
    main()
