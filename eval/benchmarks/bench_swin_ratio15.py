"""Ratio-15 three-mode benchmark tables for the Swin-B student.

Student = timm swin_base_patch4_window7_224 @112 + Linear(1024,512,bias=False)+BN1d
          (checkpoint: exp_clip_r50_vs_swin_glint_clip_ratio_15_best_model.pt)
Teacher = R50.

Partial = ratio-15 band = fixed rows [31:81] (full width) on the ArcFace-aligned 112x112
face — measured to match variants_dataset/ratio_15 AND the existing lfw/partial_15, so no
detector is needed (all benchmark faces are template-aligned). Missing <bench>_partial15
dirs are built here from the *_112x112 / cfp/full images.

Three modes per benchmark (Full<->Full / Partial->Full / Partial<->Partial), standard
10-fold protocol on the official pair lists. Standalone.
"""
import os, sys, json
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc as sk_auc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backbones import get_model

ARCHIVE = Path("/media/yoav/Yoav/datasets/benchmarks/archive/val")
CFP     = Path("/media/yoav/Yoav/datasets/benchmarks/cfp")
SWIN_CKPT = "/media/yoav/Yoav/arcface_torch_v2/exp_clip_r50_vs_swin_glint_clip_ratio_15_best_model.pt"
TEACHER   = ("work_dirs/16backbone.pth", "r50")   # full-Glint360K R50 the Swin student was distilled against
DEV, EMB, BATCH, RR = "cuda", 512, 128, 15
BAND = (31, 81)   # ratio-15 fixed band rows [y1:y2)
tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)])
FOCUS = {"LFW": "Baseline", "CALFW": "Age", "CPLFW": "Pose", "AgeDB-30": "Age (30 years)",
         "CFP-FF": "Frontal-frontal", "CFP-FP": "Frontal-profile"}


class SwinFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = timm.create_model("swin_base_patch4_window7_224", img_size=112, num_classes=0)
        self.feature = nn.Sequential(nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512))
    def forward(self, x):
        return self.feature(self.trunk(x))


def load_teacher():
    m = get_model(TEACHER[1], dropout=0.0, fp16=False, num_features=EMB).to(DEV)
    m.load_state_dict(torch.load(TEACHER[0], map_location="cpu", weights_only=True), strict=True)
    return m.eval()


def load_swin():
    m = SwinFace().to(DEV)
    m.load_state_dict(torch.load(SWIN_CKPT, map_location="cpu", weights_only=True), strict=True)
    return m.eval()


def build_partial15(src, dst):
    dst = Path(dst)
    imgs = sorted([p for p in Path(src).iterdir() if p.suffix.lower() in (".bmp", ".jpg", ".png", ".jpeg")])
    if dst.exists() and len(list(dst.glob("*"))) >= len(imgs):
        return
    dst.mkdir(parents=True, exist_ok=True)
    y1, y2 = BAND
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None: continue
        canvas = np.zeros_like(im); canvas[y1:y2, :] = im[y1:y2, :]
        cv2.imwrite(str(dst / p.name), canvas)
    print(f"  built {dst} ({len(imgs)} imgs)")


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
    flush(); return out


def metrics(scores, labels, folds):
    scores, labels, folds = map(np.asarray, (scores, labels, folds))
    grid = np.linspace(-1, 1, 4001); accs = []
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        bt = grid[int(np.argmax([((scores[tr] >= t) == labels[tr]).mean() for t in grid]))]
        accs.append(((scores[te] >= bt) == labels[te]).mean())
    accs = np.array(accs); fpr, tpr, _ = roc_curve(labels, scores)
    tar = lambda far: float(tpr[np.where(fpr <= far)[0][-1]]) if np.any(fpr <= far) else 0.0
    return {"acc": float(accs.mean()), "std": float(accs.std()), "auc": float(sk_auc(fpr, tpr)),
            "tar_1e2": tar(1e-2), "tar_1e3": tar(1e-3)}


def three_modes(pairs, fe, pe):
    Sff, Scr, Spp, y, fo = [], [], [], [], []
    for a, b, lbl, fd in pairs:
        pa, pb, fa, fb = pe[a], pe[b], fe[a], fe[b]
        Sff.append(float(fa @ fb)); Scr.append(0.5*(float(pa@fb)+float(pb@fa))); Spp.append(float(pa@pb))
        y.append(lbl); fo.append(fd)
    return {"full_full": metrics(Sff, y, fo), "partial_full": metrics(Scr, y, fo),
            "partial_partial": metrics(Spp, y, fo), "n": len(y)}


def archive_pairs(b):
    return [(Path(t.split()[1]).name, Path(t.split()[2]).name, int(t.split()[0]), i // 600)
            for i, t in enumerate((ARCHIVE / f"{b}_ann.txt").read_text().strip().splitlines())]


def cfp_pairs(proto):
    return [(f"{a}.jpg", f"{b}.jpg", lbl, fd) for a, b, lbl, fd in json.load(open(CFP / f"pairs_{proto}.json"))]


def main():
    print("Building ratio-15 partial crops (fixed band rows 31:81)...")
    for b in ["lfw", "calfw", "cplfw", "agedb_30"]:
        build_partial15(ARCHIVE / f"{b}_112x112", ARCHIVE / f"{b}_partial15")
    build_partial15(CFP / "full", CFP / "partial_15")

    teacher, swin = load_teacher(), load_swin()
    res = {}
    for disp, b in [("LFW", "lfw"), ("CALFW", "calfw"), ("CPLFW", "cplfw"), ("AgeDB-30", "agedb_30")]:
        pairs = archive_pairs(b); keys = {k for p in pairs for k in p[:2]}
        fe = embed(teacher, ARCHIVE / f"{b}_112x112", keys)
        pe = embed(swin, ARCHIVE / f"{b}_partial15", keys)
        res[disp] = three_modes(pairs, fe, pe); print(f"  [{disp}] done")
    ff, fp = cfp_pairs("FF"), cfp_pairs("FP")
    ck = {k for pl in (ff, fp) for p in pl for k in p[:2]}
    fe = embed(teacher, CFP / "full", ck); pe = embed(swin, CFP / "partial_15", ck)
    res["CFP-FF"], res["CFP-FP"] = three_modes(ff, fe, pe), three_modes(fp, fe, pe)
    print("  [CFP-FF/FP] done")

    outp = Path("work_dirs/benchmarks/swin_ratio15_three_modes.json")
    outp.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(outp, "w"), indent=2)

    order = ["LFW", "CALFW", "CPLFW", "AgeDB-30", "CFP-FF", "CFP-FP"]
    print("\n" + "=" * 92)
    print("Swin-B student, ratio_15 (rows 31:81)  —  three-mode verification accuracy % (10-fold)")
    print("=" * 92)
    print(f"{'Benchmark':<11}{'Evaluation Focus':<18}{'Full<->Full':>13}{'Partial->Full':>16}{'Partial<->Partial':>19}")
    print("-" * 92)
    for d in order:
        r = res[d]
        print(f"{d:<11}{FOCUS[d]:<18}{r['full_full']['acc']*100:>12.1f} "
              f"{r['partial_full']['acc']*100:>15.1f} {r['partial_partial']['acc']*100:>18.1f}")
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
