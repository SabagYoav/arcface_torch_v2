"""
Cross-domain verification on the pre-aligned InsightFace-style bundle
(archive/val/<bench>_112x112 + <bench>_ann.txt), for benches calfw / cplfw /
agedb_30 / lfw. Images are already ArcFace-aligned 112x112 (teacher/full domain).

For each bench: generate the clean literal-20% eye-band partial (detect eye center,
fallback to template 51.69; cached), parse the 6000-pair 10-fold protocol, and
report teacher_full_vs_full | cross_partial_vs_full | student_partial_vs_partial.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np, cv2, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

_this = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this, "..", ".."))
from backbones import get_model

ap = argparse.ArgumentParser()
ap.add_argument("--benches", nargs="+", default=["lfw", "calfw", "cplfw", "agedb_30"])
ap.add_argument("--student", default="work_dirs/exp_clip_r50_vs_vit/clip_ratio_20/best_model.pt")
ap.add_argument("--student-net", default="vit_s_dp005_mask_0")
ap.add_argument("--teacher", default="work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt")
ap.add_argument("--teacher-net", default="r50")
ap.add_argument("--ratio", type=float, default=0.20)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/media/yoav/Yoav/datasets/benchmarks/archive/val")
OUT = Path("work_dirs/benchmarks/archive"); OUT.mkdir(parents=True, exist_ok=True)
EMB, BATCH, TEMPLATE_EYE_Y = 512, 128, 51.69
tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)])


def load(net, ckpt):
    m = get_model(net, dropout=0.0, fp16=True, num_features=EMB).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True), strict=True)
    return m.eval()


def build_partial(bench, app):
    full = ROOT / f"{bench}_112x112"
    part = ROOT / f"{bench}_partial{int(round(args.ratio*100))}"
    imgs = sorted(full.glob("*.bmp"))
    if part.exists() and len(list(part.glob("*.bmp"))) == len(imgs):
        return full, part
    part.mkdir(parents=True, exist_ok=True)
    roi_h = int(round(args.ratio * 112)); nfb = 0
    for p in tqdm(imgs, desc=f"{bench} partial"):
        bgr = cv2.imread(str(p))
        faces = app.get(bgr)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            py = float((f.kps[0][1] + f.kps[1][1]) / 2.0)
        else:
            py = TEMPLATE_EYE_Y; nfb += 1
        cy = int(round(py)); y1 = max(0, cy - roi_h // 2); y2 = min(112, y1 + roi_h)
        canvas = np.zeros_like(bgr); canvas[y1:y2, :] = bgr[y1:y2, :]
        cv2.imwrite(str(part / p.name), canvas)
    print(f"  {bench}: partial built, detector-fallback={nfb}/{len(imgs)}")
    return full, part


def parse_ann(bench):
    pairs = []
    for i, ln in enumerate((ROOT / f"{bench}_ann.txt").read_text().strip().splitlines()):
        t = ln.split(); lbl = int(t[0])
        a = Path(t[1]).name; b = Path(t[2]).name
        pairs.append((a, b, lbl, i // 600))   # 6000 pairs -> 10 folds of 600
    return pairs


@torch.no_grad()
def embed(model, root, keys):
    out = {}; bi, bk = [], []
    def flush():
        if not bi: return
        e = F.normalize(model(torch.stack(bi).to(DEV)), dim=1).cpu().numpy()
        for k, v in zip(bk, e): out[k] = v
        bi.clear(); bk.clear()
    for k in tqdm(sorted(keys), desc=f"embed {root.name}"):
        bi.append(tfm(Image.open(root / k).convert("RGB"))); bk.append(k)
        if len(bi) >= BATCH: flush()
    flush(); return out


def metrics(scores, labels, folds):
    scores, labels, folds = map(np.asarray, (scores, labels, folds))
    nf = int(folds.max()) + 1; grid = np.linspace(-1, 1, 4001); accs = []
    for f in range(nf):
        tr, te = folds != f, folds == f
        bt = grid[int(np.argmax([((scores[tr] >= t) == labels[tr]).mean() for t in grid]))]
        accs.append(((scores[te] >= bt) == labels[te]).mean())
    accs = np.array(accs)
    acc_all = np.array([((scores >= t) == labels).mean() for t in grid]); gi = int(np.argmax(acc_all))
    fpr, tpr, _ = roc_curve(labels, scores); ra = auc(fpr, tpr)
    tar = lambda far: float(tpr[np.where(fpr <= far)[0][-1]]) if np.any(fpr <= far) else 0.0
    return {"acc_10fold_mean": float(accs.mean()), "acc_10fold_std": float(accs.std()),
            "best_acc": float(acc_all[gi]), "auc": float(ra),
            "tar@far=1e-2": tar(1e-2), "tar@far=1e-3": tar(1e-3)}


def main():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(160, 160))
    print(f"Student {args.student} ({args.student_net}), ratio={args.ratio}")
    student, teacher = load(args.student_net, args.student), load(args.teacher_net, args.teacher)

    summary = {}
    for bench in args.benches:
        full, part = build_partial(bench, app)
        pairs = parse_ann(bench)
        keys = set(k for p in pairs for k in p[:2])
        ft = embed(teacher, full, keys); ps = embed(student, part, keys)
        Stt, Scr, Spp, y, fo = [], [], [], [], []
        for a, b, lbl, fd in pairs:
            fa, fb, pa, pb = ft[a], ft[b], ps[a], ps[b]
            Stt.append(float(fa@fb)); Scr.append(0.5*(float(pa@fb)+float(pb@fa)))
            Spp.append(float(pa@pb)); y.append(lbl); fo.append(fd)
        summary[bench] = {"teacher_full_vs_full": metrics(Stt, y, fo),
                          "cross_partial_vs_full": metrics(Scr, y, fo),
                          "student_partial_vs_partial": metrics(Spp, y, fo)}
        print(f"[{bench}] done ({len(y)} pairs)")

    json.dump(summary, open(OUT / f"results_ratio{int(round(args.ratio*100))}.json", "w"), indent=2)
    # table
    print("\n" + "=" * 96)
    print(f"Pre-aligned benchmarks — ViT-S ratio_{int(round(args.ratio*100))} student, cross-domain (10-fold acc %)")
    print("=" * 96)
    print(f"{'bench':<12}{'teacher full/full':<20}{'CROSS partial/full':<22}{'student partial/partial':<24}{'cross AUC':<10}")
    print("-" * 96)
    for b, r in summary.items():
        t, c, s = r["teacher_full_vs_full"], r["cross_partial_vs_full"], r["student_partial_vs_partial"]
        print(f"{b:<12}{t['acc_10fold_mean']*100:6.2f}±{t['acc_10fold_std']*100:.2f}       "
              f"{c['acc_10fold_mean']*100:6.2f}±{c['acc_10fold_std']*100:.2f}         "
              f"{s['acc_10fold_mean']*100:6.2f}±{s['acc_10fold_std']*100:.2f}           {c['auc']:.4f}")
    print(f"\nSaved -> {OUT/f'results_ratio{int(round(args.ratio*100))}.json'}")


if __name__ == "__main__":
    main()
