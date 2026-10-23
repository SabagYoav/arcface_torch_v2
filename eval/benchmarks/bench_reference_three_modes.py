"""Table-VI-style three-mode verification for each BASELINE reference model.

For a single off-the-shelf model there is no teacher/student split, so the SAME model
embeds both full faces and partial crops. This measures out-of-the-box partial-face
behaviour (no distillation), in the same three modes as the paper's Table VI:

  Full  <-> Full      : cos(M(fullA),  M(fullB))
  Partial -> Full     : 0.5*(cos(M(partA), M(fullB)) + cos(M(partB), M(fullA)))
  Partial <-> Partial : cos(M(partA),  M(partB))

Standard 10-fold protocol on the official pair lists (LFW/CALFW/CPLFW/AgeDB-30/CFP-FF/CFP-FP),
partial = literal 20% eye-band crops. Reports accuracy %; JSON also stores TAR@FAR + AUC.
Standalone; only read-only `get_model`.
"""
import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc as sk_auc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backbones import get_model

ARCHIVE = Path("/media/yoav/Yoav/datasets/benchmarks/archive/val")
CFP     = Path("/media/yoav/Yoav/datasets/benchmarks/cfp")
REF     = Path("/media/yoav/Yoav/ref_checkpoints")
DEV     = "cuda" if torch.cuda.is_available() else "cpu"
EMB, BATCH, RR = 512, 128, 20
tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)])

MODELS = [
    ("InsightFace (R50)",        str(REF / "InsightFace/16backbone.pth"),             "r50"),
    ("ElasticFace (R100)",       str(REF / "ElasticFace/295672backbone.pth"),         "r100"),
    ("ElasticFaceArcAug (R100)", str(REF / "MaskInv/ElasticFaceArcAug_backbone.pth"), "r100"),
    ("MaskInv-HG (R100)",        str(REF / "MaskInv/MaskInvHG_backbone.pth"),          "r100"),
    ("MaskInv-LG (R100)",        str(REF / "MaskInv/MaskInvLG_backbone.pth"),          "r100"),
]
# (display, archive-key, focus)
BENCHES = [("LFW", "lfw", "Baseline"), ("CALFW", "calfw", "Age"), ("CPLFW", "cplfw", "Pose"),
           ("AgeDB-30", "agedb_30", "Age (30 years)")]
FOCUS = {"LFW": "Baseline", "CALFW": "Age", "CPLFW": "Pose", "AgeDB-30": "Age (30 years)",
         "CFP-FF": "Frontal-frontal", "CFP-FP": "Frontal-profile"}


def load(net, ckpt):
    m = get_model(net, dropout=0.0, fp16=False, num_features=EMB).to(DEV)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
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
    grid = np.linspace(-1, 1, 4001); accs = []
    for f in sorted(set(folds.tolist())):
        tr, te = folds != f, folds == f
        bt = grid[int(np.argmax([((scores[tr] >= t) == labels[tr]).mean() for t in grid]))]
        accs.append(((scores[te] >= bt) == labels[te]).mean())
    accs = np.array(accs)
    fpr, tpr, _ = roc_curve(labels, scores)
    tar = lambda far: float(tpr[np.where(fpr <= far)[0][-1]]) if np.any(fpr <= far) else 0.0
    return {"acc": float(accs.mean()), "std": float(accs.std()), "auc": float(sk_auc(fpr, tpr)),
            "tar_1e2": tar(1e-2), "tar_1e3": tar(1e-3)}


def three_modes(pairs, fe, pe):
    Sff, Scr, Spp, y, fo = [], [], [], [], []
    for a, b, lbl, fd in pairs:
        pa, pb, fa, fb = pe[a], pe[b], fe[a], fe[b]
        Sff.append(float(fa @ fb))
        Scr.append(0.5 * (float(pa @ fb) + float(pb @ fa)))
        Spp.append(float(pa @ pb))
        y.append(lbl); fo.append(fd)
    return {"full_full": metrics(Sff, y, fo), "partial_full": metrics(Scr, y, fo),
            "partial_partial": metrics(Spp, y, fo), "n": len(y)}


def archive_pairs(bench):
    out = []
    for i, ln in enumerate((ARCHIVE / f"{bench}_ann.txt").read_text().strip().splitlines()):
        t = ln.split(); out.append((Path(t[1]).name, Path(t[2]).name, int(t[0]), i // 600))
    return out


def cfp_pairs(proto):
    return [(f"{a}.jpg", f"{b}.jpg", lbl, fd) for a, b, lbl, fd in json.load(open(CFP / f"pairs_{proto}.json"))]


def eval_model(name, ckpt, net):
    model = load(net, ckpt)
    res = {}
    for disp, b, _ in BENCHES:
        pairs = archive_pairs(b)
        keys = {k for p in pairs for k in p[:2]}
        fe = embed(model, ARCHIVE / f"{b}_112x112", keys)
        pe = embed(model, ARCHIVE / f"{b}_partial{RR}", keys)
        res[disp] = three_modes(pairs, fe, pe)
    # CFP: embed once, score FF and FP
    ff, fp = cfp_pairs("FF"), cfp_pairs("FP")
    ckeys = {k for pl in (ff, fp) for p in pl for k in p[:2]}
    fe = embed(model, CFP / "full", ckeys)
    pe = embed(model, CFP / "partial_20", ckeys)
    res["CFP-FF"] = three_modes(ff, fe, pe)
    res["CFP-FP"] = three_modes(fp, fe, pe)
    return res


def print_table(name, res):
    order = ["LFW", "CALFW", "CPLFW", "AgeDB-30", "CFP-FF", "CFP-FP"]
    print("\n" + "=" * 92)
    print(f"{name}  —  three-mode verification accuracy % (10-fold), partial = {RR}% eye-band")
    print("=" * 92)
    print(f"{'Benchmark':<11}{'Evaluation Focus':<18}{'Full<->Full':>13}{'Partial->Full':>16}{'Partial<->Partial':>19}")
    print("-" * 92)
    for d in order:
        r = res[d]
        print(f"{d:<11}{FOCUS[d]:<18}{r['full_full']['acc']*100:>12.1f} "
              f"{r['partial_full']['acc']*100:>15.1f} {r['partial_partial']['acc']*100:>18.1f}")


def main():
    all_res = {}
    for name, ckpt, net in MODELS:
        if not Path(ckpt).exists():
            print(f"[skip] {name}: missing {ckpt}"); continue
        print(f"[running] {name} ({net}) ...", flush=True)
        all_res[name] = eval_model(name, ckpt, net)
        print_table(name, all_res[name])
    outp = Path("work_dirs/benchmarks/reference_three_modes.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_res, open(outp, "w"), indent=2)
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
