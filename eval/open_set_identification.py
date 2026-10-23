"""Open-set identification (watchlist) experiment — cross-modal.

Protocol (3000 identities total):
  * shared      (1000): enrolled in gallery AND present as probes  -> "known" probes (have a mate)
  * gallery-only(1000): enrolled in gallery, never probed          -> distractors
  * query-only  (1000): probed, NOT enrolled                       -> "unknown" probes (no mate)

  Gallery set = shared + gallery-only = 2000 templates  (full face  -> R50 TEACHER)
  Probe   set = shared + query-only   = 2000 probes     (20% partial -> ViT STUDENT)
  For shared ids the probe photo differs from the enrolled photo.

Metrics:
  * Rank@1/5/10 : CMC on the 1000 known probes (mate within top-k of the 2000-template gallery).
  * False-Alarm Rate (FAR): on the 1000 unknown probes, fraction whose top-1 gallery
    similarity exceeds a threshold (falsely flagged as enrolled), swept over thresholds.
  * DIR@rank1 @ fixed FAR: detection-and-identification rate — known probe whose mate is
    rank-1 AND above the FAR-calibrated threshold.

Standalone; only read-only `get_model`.
"""
import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backbones import get_model

DS = Path("/media/yoav/Yoav/datasets")
PART = DS / "variants_dataset/ratio_20/train"                    # partial (student domain)
FULL = DS / "glint360k/imageFolder_split_fullface/train"         # full (teacher domain)
STUDENT = ("work_dirs/exp_clip_r50_vs_vit/clip_ratio_20/best_model.pt", "vit_s_dp005_mask_0")
TEACHER = ("work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt", "r50")
EMB, BATCH, SEED, DEV = 512, 128, 2048, "cuda"
N_SHARED = N_GALLERY_ONLY = N_QUERY_ONLY = 1000
EXTS = (".jpg", ".jpeg", ".png")
tfm = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                          transforms.Normalize([0.5]*3, [0.5]*3)])


def load(ckpt, net):
    m = get_model(net, dropout=0.0, fp16=False, num_features=EMB).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    return m.eval()


def imgs_of(root, cid):
    return sorted(p.name for p in (root / cid).iterdir() if p.suffix.lower() in EXTS)


@torch.no_grad()
def embed_paths(model, paths):
    embs, bi = [], []
    def flush():
        if not bi: return
        embs.append(F.normalize(model(torch.stack(bi).to(DEV)).float(), dim=1).cpu())
        bi.clear()
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

    # shared: need >=2 partial imgs (probe photo != enrolled photo)
    shared = []
    it = iter(common)
    for cid in it:
        if len(imgs_of(PART, cid)) >= 2:
            shared.append(cid)
        if len(shared) == N_SHARED:
            break
    used = set(shared)
    query_only = [c for c in common if c not in used][:N_QUERY_ONLY]          # partial probes, not enrolled
    used |= set(query_only)
    gallery_only = [c for c in sorted(full_ids, key=int) if c not in used][:N_GALLERY_ONLY]  # full distractors
    assert len(shared) == 1000 and len(query_only) == 1000 and len(gallery_only) == 1000
    print(f"ids: shared={len(shared)} gallery_only={len(gallery_only)} query_only={len(query_only)}")

    # ---- build gallery (full -> teacher) ----
    gal_ids, gal_paths, gal_gphoto = [], [], {}
    for cid in shared + gallery_only:
        photo = imgs_of(FULL, cid)[0]
        gal_gphoto[cid] = photo
        gal_ids.append(cid); gal_paths.append(FULL / cid / photo)
    gal_index = {cid: i for i, cid in enumerate(gal_ids)}   # id -> gallery row

    # ---- build probes (partial -> student) ----
    prb_ids, prb_paths, prb_known = [], [], []
    for cid in shared:
        cand = [n for n in imgs_of(PART, cid) if n != gal_gphoto[cid]]        # different photo
        photo = cand[0] if cand else imgs_of(PART, cid)[0]
        prb_ids.append(cid); prb_paths.append(PART / cid / photo); prb_known.append(True)
    for cid in query_only:
        photo = imgs_of(PART, cid)[0]
        prb_ids.append(cid); prb_paths.append(PART / cid / photo); prb_known.append(False)
    prb_known = np.array(prb_known)

    teacher, student = load(*TEACHER), load(*STUDENT)
    print("embedding gallery (teacher, full)...");  G = embed_paths(teacher, gal_paths)   # [2000,512]
    print("embedding probes (student, partial)..."); Q = embed_paths(student, prb_paths)  # [2000,512]

    S = Q @ G.T                                   # [2000 probes, 2000 gallery] cosine
    order = np.argsort(-S, axis=1)                # gallery indices sorted by sim desc
    top1_sim = S[np.arange(len(S)), order[:, 0]]

    # ---- CMC on known probes ----
    known_idx = np.where(prb_known)[0]
    mate_row = np.array([gal_index[prb_ids[i]] for i in known_idx])
    ranks = np.array([np.where(order[i] == mate_row[j])[0][0] for j, i in enumerate(known_idx)])
    cmc = {k: float((ranks < k).mean()) for k in (1, 5, 10)}

    # ---- open set: FAR on unknown, DIR@1 on known ----
    unk_idx = np.where(~prb_known)[0]
    unk_top1 = top1_sim[unk_idx]
    known_top1 = top1_sim[known_idx]
    known_rank1_correct = (ranks == 0)

    def far_at(tau):  return float((unk_top1 >= tau).mean())
    def dir1_at(tau): return float((known_rank1_correct & (known_top1 >= tau)).mean())
    # thresholds calibrated to target FAR on unknown probes
    ops = {}
    for far_t in (0.10, 0.01, 0.001):
        tau = float(np.quantile(unk_top1, 1 - far_t))
        ops[far_t] = {"threshold": tau, "FAR": far_at(tau), "DIR@rank1": dir1_at(tau)}

    tau_sweep = [round(t, 2) for t in np.arange(0.0, 0.71, 0.1)]
    far_curve = {t: far_at(t) for t in tau_sweep}

    res = {"counts": {"gallery": len(gal_ids), "probes": len(prb_ids),
                      "known": int(prb_known.sum()), "unknown": int((~prb_known).sum())},
           "CMC": cmc, "operating_points": ops, "FAR_sweep": far_curve}
    out = Path("work_dirs/benchmarks/open_set_identification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)

    # ---- report ----
    print("\n" + "=" * 70)
    print("OPEN-SET IDENTIFICATION  (partial probe -> ViT student ; full gallery -> R50 teacher)")
    print("=" * 70)
    print(f"gallery templates : {len(gal_ids)}  (1000 mates + 1000 distractors)")
    print(f"probes            : {len(prb_ids)}  (1000 known + 1000 unknown)")
    print("\nClosed-set CMC on known probes:")
    print(f"  Rank@1 = {cmc[1]*100:.2f}%   Rank@5 = {cmc[5]*100:.2f}%   Rank@10 = {cmc[10]*100:.2f}%")
    print("\nOpen-set operating points (threshold calibrated to target FAR on unknown probes):")
    print(f"  {'target FAR':>10} | {'threshold':>9} | {'actual FAR':>10} | {'DIR@rank1':>9}")
    for far_t, o in ops.items():
        print(f"  {far_t*100:>9.1f}% | {o['threshold']:>9.3f} | {o['FAR']*100:>9.2f}% | {o['DIR@rank1']*100:>8.2f}%")
    print("\nFalse-Alarm Rate vs threshold (unknown probes flagged as enrolled):")
    print("  " + "  ".join(f"τ={t:.1f}:{far_curve[t]*100:5.1f}%" for t in tau_sweep))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
