"""Standard-protocol cross-modal verification across ALL benchmark datasets.

STANDALONE — reuses only read-only `get_model`; touches no training code, and
needs no face detector / network (partial crops are already built on disk).

For every benchmark (LFW, CALFW, CPLFW, AgeDB-30, CFP-FF, CFP-FP):
  probe     = partial-face crop -> STUDENT  (exp_clip_r50_vs_vit, ViT-S)
  reference = full face         -> TEACHER  (R50)
  pair score (cross-modal, symmetric) = 0.5*(cos(pA_partial, pB_full) + cos(pB_partial, pA_full))

Standard protocol on the official pair lists:
  * Accuracy : k-fold CV — threshold fit on train folds, applied to held-out fold
               (LFW-style, no leakage); reported mean ± std.
  * TAR@FAR  : at FAR = 1e-1, 1e-2, 1e-3 from the global ROC.
  * AUC.
Also reports teacher full/full as a sanity anchor.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc as sk_auc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backbones import get_model  # read-only

ARCHIVE = Path("/media/yoav/Yoav/datasets/benchmarks/archive/val")
CFP     = Path("/media/yoav/Yoav/datasets/benchmarks/cfp")
DEV     = "cuda" if torch.cuda.is_available() else "cpu"
EMB, BATCH = 512, 128
tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)])

ap = argparse.ArgumentParser()
ap.add_argument("--student", default="work_dirs/exp_clip_r50_vs_vit/clip_ratio_20/best_model.pt")
ap.add_argument("--student-net", default="vit_s_dp005_mask_0")
ap.add_argument("--teacher", default="work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt")
ap.add_argument("--teacher-net", default="r50")
ap.add_argument("--ratio", type=int, default=20)   # partial-crop percentage on disk
args = ap.parse_args()


def load(net, ckpt):
    m = get_model(net, dropout=0.0, fp16=False, num_features=EMB).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    return m.eval()


@torch.no_grad()
def embed(model, root, keys):
    out, bi, bk = {}, [], []
    def flush():
        if not bi: return
        e = F.normalize(model(torch.stack(bi).to(DEV)).float(), dim=1).cpu().numpy()
        for k, v in zip(bk, e): out[k] = v
        bi.clear(); bk.clear()
    for k in sorted(keys):
        bi.append(tfm(Image.open(root / k).convert("RGB"))); bk.append(k)
        if len(bi) >= BATCH: flush()
    flush()
    return out


def metrics(scores, labels, folds):
    scores, labels, folds = map(np.asarray, (scores, labels, folds))
    grid = np.linspace(-1, 1, 4001)
    accs = []
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        bt = grid[int(np.argmax([((scores[tr] >= t) == labels[tr]).mean() for t in grid]))]
        accs.append(((scores[te] >= bt) == labels[te]).mean())
    accs = np.array(accs)
    fpr, tpr, _ = roc_curve(labels, scores)
    tar = lambda far: float(tpr[np.where(fpr <= far)[0][-1]]) if np.any(fpr <= far) else 0.0
    return {"acc": float(accs.mean()), "std": float(accs.std()), "auc": float(sk_auc(fpr, tpr)),
            "tar_1e1": tar(1e-1), "tar_1e2": tar(1e-2), "tar_1e3": tar(1e-3)}


def archive_pairs(bench):
    pairs = []
    for i, ln in enumerate((ARCHIVE / f"{bench}_ann.txt").read_text().strip().splitlines()):
        t = ln.split()
        pairs.append((Path(t[1]).name, Path(t[2]).name, int(t[0]), i // 600))
    return pairs


def score_pairs(pairs, full_emb, part_emb):
    Sff, Scr, Spp, y, fo = [], [], [], [], []
    for a, b, lbl, fd in pairs:
        pa, pb, fa, fb = part_emb[a], part_emb[b], full_emb[a], full_emb[b]
        Sff.append(float(fa @ fb))                             # full <-> full   (teacher/teacher)
        Scr.append(0.5 * (float(pa @ fb) + float(pb @ fa)))    # partial <-> full (student/teacher)
        Spp.append(float(pa @ pb))                             # partial <-> partial (student/student)
        y.append(lbl); fo.append(fd)
    return Sff, Scr, Spp, y, fo


def three_modes(pairs, full_emb, part_emb):
    Sff, Scr, Spp, y, fo = score_pairs(pairs, full_emb, part_emb)
    return {"full_full": metrics(Sff, y, fo),
            "partial_full": metrics(Scr, y, fo),
            "partial_partial": metrics(Spp, y, fo),
            "n": len(y)}


def main():
    print(f"Student: {args.student} ({args.student_net}) | ratio_{args.ratio}")
    student, teacher = load(args.student_net, args.student), load(args.teacher_net, args.teacher)
    RR = args.ratio
    results = {}

    # ---- archive benches ----
    for name, b in [("LFW", "lfw"), ("CALFW", "calfw"), ("CPLFW", "cplfw"), ("AgeDB-30", "agedb_30")]:
        full_dir, part_dir = ARCHIVE / f"{b}_112x112", ARCHIVE / f"{b}_partial{RR}"
        pairs = archive_pairs(b)
        keys = {k for p in pairs for k in p[:2]}
        fe, pe = embed(teacher, full_dir, keys), embed(student, part_dir, keys)
        results[name] = three_modes(pairs, fe, pe)
        print(f"  [{name}] {results[name]['n']} pairs done")

    # ---- CFP (embed the 7000 images once, score FF and FP) ----
    cfp_pairs = {p.replace("pairs_", "").replace(".json", ""): json.load(open(CFP / p))
                 for p in ["pairs_FF.json", "pairs_FP.json"]}
    cfp_keys = {f"{k}.jpg" for pl in cfp_pairs.values() for a, b, _, _ in pl for k in (a, b)}
    fe = embed(teacher, CFP / "full", cfp_keys)
    pe = embed(student, CFP / "partial_20", cfp_keys)
    for proto, pl in cfp_pairs.items():
        pairs = [(f"{a}.jpg", f"{b}.jpg", lbl, fd) for a, b, lbl, fd in pl]
        results[f"CFP-{proto}"] = three_modes(pairs, fe, pe)
        print(f"  [CFP-{proto}] {results[f'CFP-{proto}']['n']} pairs done")

    outp = Path("work_dirs/benchmarks") / f"standard_bench_vit_ratio{RR}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outp, "w"), indent=2)

    # ---- table (per dataset: full/full, partial/full, partial/partial) ----
    print(f"\nSTANDARD-PROTOCOL verification — ViT-S ratio_{RR} student + R50 teacher  (10-fold acc, TAR@FAR, AUC)")
    modes = [("full_full", "full <-> full     (teacher/teacher)"),
             ("partial_full", "partial <-> full  (student/teacher)"),
             ("partial_partial", "partial <-> partial (student/student)")]
    for name, r in results.items():
        print("\n" + "=" * 92)
        print(f"{name}   ({r['n']} pairs)")
        print("-" * 92)
        print(f"  {'mode':<38}| {'Acc(10f)':>13} | {'TAR@1e-1':>8} | {'TAR@1e-2':>8} | {'TAR@1e-3':>8} | {'AUC':>6}")
        print("  " + "-" * 90)
        for key, label in modes:
            m = r[key]
            print(f"  {label:<38}| {m['acc']*100:>6.2f}±{m['std']*100:>4.2f}% | "
                  f"{m['tar_1e1']*100:>7.2f}% | {m['tar_1e2']*100:>7.2f}% | {m['tar_1e3']*100:>7.2f}% | {m['auc']:>6.4f}")
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
