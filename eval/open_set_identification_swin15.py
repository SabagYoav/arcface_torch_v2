"""Open-set identification (watchlist) — 15% partial probe, Swin-B student.

Same protocol as eval/open_set_identification.py but:
  probe     = ratio_15 partial crop -> Swin-B student (exp_clip_r50_vs_swin_glint_clip_ratio_15)
  reference = full face             -> R50 teacher (16backbone)

3000 identities: shared(1000, known probes) + gallery_only(1000, distractors) + query_only(1000, unknown probes).
Gallery = 2000 templates (full -> teacher). Probes = 2000 (partial -> student); for shared ids the probe
photo differs from the enrolled photo. Metrics: Rank@1/5/10 (CMC on known), FAR (unknown), DIR@rank1.
"""
import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backbones import get_model

DS = Path("/media/yoav/Yoav/datasets")
PART = DS / "variants_dataset/ratio_15/train"                    # 15% partial (student domain)
FULL = DS / "glint360k/imageFolder_split_fullface/train"         # full (teacher domain)
SWIN_CKPT = "/media/yoav/Yoav/arcface_torch_v2/exp_clip_r50_vs_swin_glint_clip_ratio_15_best_model.pt"
TEACHER_CKPT = "work_dirs/16backbone.pth"
EMB, BATCH, SEED, DEV = 512, 128, 2048, "cuda"
N_SHARED = N_GALLERY_ONLY = N_QUERY_ONLY = 1000
EXTS = (".jpg", ".jpeg", ".png")
tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)])


class SwinFace(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = timm.create_model("swin_base_patch4_window7_224", img_size=112, num_classes=0)
        self.feature = nn.Sequential(nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512))
    def forward(self, x):
        return self.feature(self.trunk(x))


def load_swin():
    m = SwinFace().to(DEV)
    m.load_state_dict(torch.load(SWIN_CKPT, map_location="cpu", weights_only=True), strict=True)
    return m.eval()


def load_teacher():
    m = get_model("r50", dropout=0.0, fp16=False, num_features=EMB).to(DEV)
    m.load_state_dict(torch.load(TEACHER_CKPT, map_location="cpu", weights_only=True), strict=True)
    return m.eval()


def imgs_of(root, cid):
    return sorted(p.name for p in (root / cid).iterdir() if p.suffix.lower() in EXTS)


@torch.no_grad()
def embed_paths(model, paths):
    embs, bi = [], []
    def flush():
        if not bi: return
        embs.append(F.normalize(model(torch.stack(bi).to(DEV)).float(), dim=1).cpu()); bi.clear()
    for p in paths:
        bi.append(tfm(Image.open(p).convert("RGB")))
        if len(bi) >= BATCH: flush()
    flush()
    return torch.cat(embs).numpy()


def main():
    rng = np.random.default_rng(SEED)
    part_ids = {d.name for d in PART.iterdir() if d.is_dir() and d.name.isdigit()}
    full_ids = {d.name for d in FULL.iterdir() if d.is_dir() and d.name.isdigit()}
    common = sorted(part_ids & full_ids, key=int)
    rng.shuffle(common)

    shared = []
    for cid in common:
        if len(imgs_of(PART, cid)) >= 2:
            shared.append(cid)
        if len(shared) == N_SHARED:
            break
    used = set(shared)
    query_only = [c for c in common if c not in used][:N_QUERY_ONLY]
    used |= set(query_only)
    gallery_only = [c for c in sorted(full_ids, key=int) if c not in used][:N_GALLERY_ONLY]
    assert len(shared) == 1000 and len(query_only) == 1000 and len(gallery_only) == 1000
    print(f"ids: shared={len(shared)} gallery_only={len(gallery_only)} query_only={len(query_only)}")

    gal_ids, gal_paths, gal_gphoto = [], [], {}
    for cid in shared + gallery_only:
        photo = imgs_of(FULL, cid)[0]
        gal_gphoto[cid] = photo
        gal_ids.append(cid); gal_paths.append(FULL / cid / photo)
    gal_index = {cid: i for i, cid in enumerate(gal_ids)}

    prb_ids, prb_paths, prb_known = [], [], []
    for cid in shared:
        cand = [n for n in imgs_of(PART, cid) if n != gal_gphoto[cid]]
        photo = cand[0] if cand else imgs_of(PART, cid)[0]
        prb_ids.append(cid); prb_paths.append(PART / cid / photo); prb_known.append(True)
    for cid in query_only:
        prb_ids.append(cid); prb_paths.append(PART / cid / imgs_of(PART, cid)[0]); prb_known.append(False)
    prb_known = np.array(prb_known)

    teacher, student = load_teacher(), load_swin()
    print("embedding gallery (teacher, full)...");  G = embed_paths(teacher, gal_paths)
    print("embedding probes (student, 15% partial)..."); Q = embed_paths(student, prb_paths)

    S = Q @ G.T
    order = np.argsort(-S, axis=1)
    top1_sim = S[np.arange(len(S)), order[:, 0]]

    known_idx = np.where(prb_known)[0]
    mate_row = np.array([gal_index[prb_ids[i]] for i in known_idx])
    ranks = np.array([np.where(order[i] == mate_row[j])[0][0] for j, i in enumerate(known_idx)])
    cmc = {k: float((ranks < k).mean()) for k in (1, 5, 10)}

    unk_idx = np.where(~prb_known)[0]
    unk_top1 = top1_sim[unk_idx]; known_top1 = top1_sim[known_idx]; r1 = (ranks == 0)
    far_at = lambda t: float((unk_top1 >= t).mean())
    dir1_at = lambda t: float((r1 & (known_top1 >= t)).mean())
    ops = {}
    for f in (0.10, 0.01, 0.001):
        tau = float(np.quantile(unk_top1, 1 - f))
        ops[f] = {"threshold": tau, "FAR": far_at(tau), "DIR@rank1": dir1_at(tau)}
    tau_sweep = [round(t, 2) for t in np.arange(0.0, 0.71, 0.1)]
    far_curve = {t: far_at(t) for t in tau_sweep}

    res = {"probe": "ratio_15 (Swin-B student)", "gallery": "full (R50 teacher 16backbone)",
           "counts": {"gallery": len(gal_ids), "probes": len(prb_ids), "known": 1000, "unknown": 1000},
           "CMC": cmc, "operating_points": ops, "FAR_sweep": far_curve}
    out = Path("work_dirs/benchmarks/open_set_identification_swin15.json")
    out.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"), indent=2)

    print("\n" + "=" * 72)
    print("OPEN-SET IDENTIFICATION  (15% partial probe -> Swin-B ; full gallery -> R50 teacher)")
    print("=" * 72)
    print(f"gallery templates : {len(gal_ids)} (1000 mates + 1000 distractors)")
    print(f"probes            : {len(prb_ids)} (1000 known + 1000 unknown)")
    print("\nClosed-set CMC on known probes:")
    print(f"  Rank@1 = {cmc[1]*100:.2f}%   Rank@5 = {cmc[5]*100:.2f}%   Rank@10 = {cmc[10]*100:.2f}%")
    print("\nOpen-set operating points (threshold calibrated to target FAR on unknown probes):")
    print(f"  {'target FAR':>10} | {'threshold':>9} | {'actual FAR':>10} | {'DIR@rank1':>9}")
    for f, o in ops.items():
        print(f"  {f*100:>9.1f}% | {o['threshold']:>9.3f} | {o['FAR']*100:>9.2f}% | {o['DIR@rank1']*100:>8.2f}%")
    print("\nFalse-Alarm Rate vs threshold:")
    print("  " + "  ".join(f"τ={t:.1f}:{far_curve[t]*100:5.1f}%" for t in tau_sweep))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
