"""RFW (Racial Faces in the Wild) three-mode benchmark, ratio-15 band.

Evaluates, per racial subset (Caucasian / African / Asian / Indian), five models:
  * Swin-B ratio_15 student (ours)  -> Partial encoder = Swin, Full encoder = R50 teacher (16backbone)
  * InsightFace (R50), ElasticFace (R100), MaskInv-HG (R100), MaskInv-LG (R100)  -> single model both roles

Three modes (standard 10-fold protocol on the official 6000-pair .bin packs):
  Full<->Full | Partial->Full (0.5*(cos(pA,fB)+cos(pB,fA))) | Partial<->Partial.
Partial = ratio-15 fixed band rows [31:81] on the aligned 112x112 face (same for all models
=> apples-to-apples). Standalone; reads .bin packs directly (no mxnet).
"""
import os, sys, io, json, pickle
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc as sk_auc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backbones import get_model

RFW = Path("/media/yoav/Yoav/datasets/benchmarks/RFW/bin_for_mxnet/RFW_test")
REF = Path("/media/yoav/Yoav/ref_checkpoints")
SWIN_CKPT = "/media/yoav/Yoav/arcface_torch_v2/exp_clip_r50_vs_swin_glint_clip_ratio_15_best_model.pt"
R50_TEACHER = "work_dirs/16backbone.pth"
DEV, EMB, BATCH, BAND = "cuda", 512, 128, (31, 81)
RACES = ["Caucasian", "African", "Asian", "Indian"]
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])


class SwinFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = timm.create_model("swin_base_patch4_window7_224", img_size=112, num_classes=0)
        self.feature = nn.Sequential(nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512))
    def forward(self, x):
        return self.feature(self.trunk(x))


def load(net, ckpt):
    m = SwinFace() if net == "swin" else get_model(net, dropout=0.0, fp16=False, num_features=EMB)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    return m.to(DEV).eval()


def load_bin(path):
    bins, issame = pickle.load(open(path, "rb"), encoding="bytes")
    imgs = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in bins]
    return imgs, np.asarray(issame, dtype=bool)


def band(im):
    c = np.zeros_like(im); y1, y2 = BAND; c[y1:y2, :] = im[y1:y2, :]; return c


@torch.no_grad()
def embed(model, arrays):
    out, bi = [], []
    def flush():
        if not bi: return
        out.append(F.normalize(model(torch.stack(bi).to(DEV)).float(), dim=1).cpu().numpy()); bi.clear()
    for a in arrays:
        bi.append(tfm(Image.fromarray(a)))
        if len(bi) >= BATCH: flush()
    flush()
    return np.concatenate(out)


def metrics(S, y, fo):
    S, y, fo = map(np.asarray, (S, y, fo)); grid = np.linspace(-1, 1, 4001); accs = []
    for f in sorted(set(fo.tolist())):
        tr, te = fo != f, fo == f
        bt = grid[int(np.argmax([((S[tr] >= t) == y[tr]).mean() for t in grid]))]
        accs.append(((S[te] >= bt) == y[te]).mean())
    fpr, tpr, _ = roc_curve(y, S)
    tar = lambda far: float(tpr[np.where(fpr <= far)[0][-1]]) if np.any(fpr <= far) else 0.0
    return {"acc": float(np.mean(accs)), "std": float(np.std(accs)), "auc": float(sk_auc(fpr, tpr)),
            "tar_1e2": tar(1e-2), "tar_1e3": tar(1e-3)}


def modes(part_enc, full_enc, E, pairs):
    y = [p[2] for p in pairs]; fo = [p[3] for p in pairs]
    Ff, Pf, Pp = E[full_enc]["f"], E[part_enc]["p"], E[full_enc]["f"]
    FF = [float(Ff[a] @ Ff[b]) for a, b, _, _ in pairs]
    PF = [0.5*(float(E[part_enc]["p"][a] @ Ff[b]) + float(E[part_enc]["p"][b] @ Ff[a])) for a, b, _, _ in pairs]
    PP = [float(E[part_enc]["p"][a] @ E[part_enc]["p"][b]) for a, b, _, _ in pairs]
    return {"full_full": metrics(FF, y, fo), "partial_full": metrics(PF, y, fo), "partial_partial": metrics(PP, y, fo)}


# model -> (partial_encoder, full_encoder) keys into the encoder dict
ENCODERS = {
    "teacher_r50": ("r50",  R50_TEACHER),
    "swin":        ("swin", SWIN_CKPT),
    "elastic":     ("r100", str(REF / "ElasticFace/295672backbone.pth")),
    "maskhg":      ("r100", str(REF / "MaskInv/MaskInvHG_backbone.pth")),
    "masklg":      ("r100", str(REF / "MaskInv/MaskInvLG_backbone.pth")),
}
MODEL_MAP = [  # (display, partial_enc, full_enc)
    ("Swin-B ratio_15 (ours)", "swin",        "teacher_r50"),
    ("InsightFace (R50)",       "teacher_r50", "teacher_r50"),
    ("ElasticFace (R100)",      "elastic",     "elastic"),
    ("MaskInv-HG (R100)",       "maskhg",      "maskhg"),
    ("MaskInv-LG (R100)",       "masklg",      "masklg"),
]


def main():
    print("loading encoders...")
    enc = {k: load(net, ck) for k, (net, ck) in ENCODERS.items()}
    results = {disp: {} for disp, _, _ in MODEL_MAP}
    for race in RACES:
        imgs, issame = load_bin(RFW / f"{race}_test.bin")
        parts = [band(im) for im in imgs]
        pairs = [(2*i, 2*i+1, bool(issame[i]), i // 600) for i in range(len(issame))]
        print(f"[{race}] {len(issame)} pairs; embedding {len(imgs)} imgs x {len(enc)} encoders...")
        E = {k: {"f": embed(m, imgs), "p": embed(m, parts)} for k, m in enc.items()}
        for disp, pe, fe in MODEL_MAP:
            results[disp][race] = modes(pe, fe, E, pairs)
        print(f"[{race}] done")

    outp = Path("work_dirs/benchmarks/rfw_ratio15_three_modes.json")
    outp.parent.mkdir(parents=True, exist_ok=True); json.dump(results, open(outp, "w"), indent=2)

    for disp, _, _ in MODEL_MAP:
        print("\n" + "=" * 80)
        print(f"{disp}  —  RFW three-mode verification accuracy % (10-fold), ratio-15 band")
        print("-" * 80)
        print(f"{'Race':<12}{'Full<->Full':>13}{'Partial->Full':>16}{'Partial<->Partial':>19}")
        for race in RACES:
            r = results[disp][race]
            print(f"{race:<12}{r['full_full']['acc']*100:>12.1f} "
                  f"{r['partial_full']['acc']*100:>15.1f} {r['partial_partial']['acc']*100:>18.1f}")
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
