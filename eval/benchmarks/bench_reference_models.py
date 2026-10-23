"""Standard full-face verification benchmark for reference / SOTA checkpoints.

Evaluates each reference model (full <-> full, both images embedded by the SAME model)
on LFW / CALFW / CPLFW / AgeDB-30 / CFP-FF / CFP-FP under the standard 10-fold protocol
(official pair lists, threshold fit on train folds). Reports 10-fold accuracy; the JSON
also stores TAR@FAR and AUC. Our R50 teacher is included as a reference row.

Standalone: only read-only `get_model`; no training code touched.
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
EMB, BATCH = 512, 128
tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)])

# (display name, checkpoint, backbone)
MODELS = [
    ("Ours-Teacher (R50)",     "work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt", "r50"),
    ("InsightFace (R50)",       str(REF / "InsightFace/16backbone.pth"),                                  "r50"),
    ("ElasticFace (R100)",      str(REF / "ElasticFace/295672backbone.pth"),                              "r100"),
    ("ElasticFaceArcAug (R100)",str(REF / "MaskInv/ElasticFaceArcAug_backbone.pth"),                      "r100"),
    ("MaskInv-HG (R100)",       str(REF / "MaskInv/MaskInvHG_backbone.pth"),                              "r100"),
    ("MaskInv-LG (R100)",       str(REF / "MaskInv/MaskInvLG_backbone.pth"),                              "r100"),
]
BENCHES = [("LFW", "lfw"), ("CALFW", "calfw"), ("CPLFW", "cplfw"), ("AgeDB-30", "agedb_30")]


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
            "tar_1e2": tar(1e-2), "tar_1e3": tar(1e-3)}


def archive_pairs(bench):
    out = []
    for i, ln in enumerate((ARCHIVE / f"{bench}_ann.txt").read_text().strip().splitlines()):
        t = ln.split()
        out.append((Path(t[1]).name, Path(t[2]).name, int(t[0]), i // 600))
    return out


def cfp_pairs(proto):
    return [(f"{a}.jpg", f"{b}.jpg", lbl, fd) for a, b, lbl, fd in json.load(open(CFP / f"pairs_{proto}.json"))]


# (name, full_dir, pairs)
DATASETS = [(n, ARCHIVE / f"{b}_112x112", archive_pairs(b)) for n, b in BENCHES] + \
           [("CFP-FF", CFP / "full", cfp_pairs("FF")), ("CFP-FP", CFP / "full", cfp_pairs("FP"))]


def main():
    results = {}
    for name, ckpt, net in MODELS:
        if not Path(ckpt).exists():
            print(f"[skip] {name}: missing {ckpt}"); continue
        model = load(net, ckpt)
        results[name] = {}
        for ds_name, full_dir, pairs in DATASETS:
            keys = {k for p in pairs for k in p[:2]}
            fe = embed(model, full_dir, keys)
            S = [float(fe[a] @ fe[b]) for a, b, _, _ in pairs]
            y = [p[2] for p in pairs]; fo = [p[3] for p in pairs]
            results[name][ds_name] = metrics(S, y, fo)
        print(f"[done] {name}: " + "  ".join(f"{d}={results[name][d]['acc']*100:.2f}" for d, *_ in DATASETS))

    outp = Path("work_dirs/benchmarks/reference_models_fullface.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outp, "w"), indent=2)

    cols = [d for d, *_ in DATASETS]
    print("\n" + "=" * 104)
    print("REFERENCE MODELS — standard full<->full verification accuracy % (10-fold)")
    print("=" * 104)
    hdr = f"{'Model':<26}| " + " | ".join(f"{c:>8}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for name in results:
        print(f"{name:<26}| " + " | ".join(f"{results[name][c]['acc']*100:>7.2f}%" for c in cols))
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
