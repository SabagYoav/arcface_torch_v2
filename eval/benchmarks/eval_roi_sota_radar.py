"""Partial (ROI 0.15) vs full verification: our swin_base vs public face models,
over six benchmarks (LFW, CALFW, CPLFW, AgeDB-30, CFP-FF, CFP-FP), and a
radar chart of the 10-fold mean accuracies.

Same protocol and code as eval_roi_loss_ablation.py (buffalo_l eye centers +
crop_roi, symmetric pair score, 10-fold acc on the official folds):
  * Swin-B (ours): CLIP-distilled swin_base, full glint360k, 20 epochs
    (exp_clip_r50_vs_swin_glint/clip_ratio_15, final epoch). Probe = ROI crop ->
    swin_base, reference = full face -> its R50 teacher (the deployed pairing).
  * ElasticFace-Arc (R100, MS1MV2), MaskInv-HG (R100, MS1MV2; Huber et al.),
    InsightFace R50 (16backbone.pth -- also the swin student's teacher):
    off-the-shelf from work_dirs/ref_checkpoints, the SAME model embeds both
    the ROI probe and the full reference.
Third-party .pth weights are loaded with weights_only=True.

Each model is scored on three tasks: full-vs-full (reference model on both
sides), partial-vs-full (the radar), partial-vs-partial (probe model on both).

Usage (in the devcontainer, onnxruntime-gpu needs torch's cuDNN on LD_LIBRARY_PATH):
  python3 eval/benchmarks/eval_roi_sota_radar.py           # evaluate + plot
  python3 eval/benchmarks/eval_roi_sota_radar.py --plot    # re-plot from results.json
  python3 eval/benchmarks/eval_roi_sota_radar.py --suite rfw   # RFW (4 races) -> results_rfw.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from eval.benchmarks.eval_roi_loss_ablation import (
    BENCH, RATIO, cross_scores, embed, eye_centers, load_cfp_fp, load_lfw, load_model,
    to_tensors, verification_metrics,
)
from eval.benchmarks.lfw_bin import compute_or_load_eye_centers, decode_lfw_bin, pair_folds
from utils.clip_verifications_utils import ClipVerification

OUT = Path("work_dirs/benchmarks/roi_sota_radar")
WEIGHTS = Path("work_dirs/ref_checkpoints")
TEACHER = ("r50", "16backbone.pth")
BENCHES = ["LFW", "CALFW", "CPLFW", "AgeDB-30", "CFP-FF", "CFP-FP"]
RFW = ["African", "Asian", "Caucasian", "Indian"]
# Fixed categorical order (dataviz reference palette, light mode, slots 1-4).
MODELS = {
    "Swin-B (ours)": {"color": "#2a78d6"},
    "ElasticFace-Arc": {"color": "#eb6834"},
    "MaskInv-HG": {"color": "#1baf7a"},
    "InsightFace (R50)": {"color": "#eda100"},
}


def load_archive(name):
    root = BENCH / "val"
    labels, paths = [], []
    for line in open(root / f"{name}_ann.txt"):
        lab, a, b = line.split()
        labels.append(lab == "1")
        paths += [root / a, root / b]
    return [cv2.imread(str(p)) for p in paths], np.array(labels), pair_folds(len(labels))


def load_cfp_ff():
    root = BENCH / "cfp"
    pairs = json.load(open(root / "pairs_FF.json"))
    images = []
    for a, b, _, _ in pairs:
        images += [cv2.imread(str(root / "full" / f"{a}.jpg")), cv2.imread(str(root / "full" / f"{b}.jpg"))]
    return images, np.array([p[2] == 1 for p in pairs]), np.array([p[3] for p in pairs])


def load_rfw(race):
    images, issame = decode_lfw_bin(BENCH / f"RFW/bin_for_mxnet/RFW_test/{race}_test.bin")
    return images, np.array(issame, dtype=bool), pair_folds(len(issame))


LOADERS = {"LFW": load_lfw, "CALFW": lambda: load_archive("calfw"), "CPLFW": lambda: load_archive("cplfw"),
           "AgeDB-30": lambda: load_archive("agedb_30"), "CFP-FF": load_cfp_ff, "CFP-FP": load_cfp_fp,
           **{f"RFW-{r}": (lambda r=r: load_rfw(r)) for r in RFW}}


class OnnxEmbedder:
    """InsightFace ONNX recognizer behind the same call interface as a torch
    backbone. Its own preprocessing is RGB, (x - 127.5) / 127.5 -- identical to
    our transform's Normalize(0.5, 0.5) on [0, 1] RGB tensors."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, batch):
        out = self.sess.run(None, {self.inp: batch.float().cpu().numpy()})[0]
        return torch.from_numpy(out).cuda()

    def eval(self):
        return self


def load_r100(path):
    from backbones import get_model
    m = get_model("r100", dropout=0.0, fp16=True, num_features=512).cuda()
    m.load_state_dict(torch.load(path, map_location="cuda", weights_only=True))
    return m.eval()


def evaluate(benches, out_name):
    OUT.mkdir(parents=True, exist_ok=True)
    app = ClipVerification._build_detector()
    data = {}
    for name in benches:
        images, labels, folds = LOADERS[name]()
        assert all(im is not None for im in images), f"{name}: unreadable image"
        if name == "LFW":
            centers = [c["center_y"] for c in compute_or_load_eye_centers(images, app)]
        else:
            centers = eye_centers(name, images, app)
        data[name] = {"full": to_tensors(images), "partial": to_tensors(images, centers),
                      "labels": labels, "folds": folds}
        print(f"{name}: {len(labels)} pairs", flush=True)

    def run(partial_model, full_model):
        r = {}
        for n, d in data.items():
            Ef = embed(full_model, d["full"])
            Ep = embed(partial_model, d["partial"])
            m = lambda A, B: verification_metrics(cross_scores(A, B), d["labels"], d["folds"])
            r[n] = {"full_vs_full": m(Ef, Ef), "partial_vs_full": m(Ep, Ef), "partial_vs_partial": m(Ep, Ep)}
        return r

    results = {}
    teacher = load_model(*TEACHER)
    swin = load_model("swin_base", "work_dirs/exp_clip_r50_vs_swin_glint/clip_ratio_15/model.pt")
    results["Swin-B (ours)"] = run(swin, teacher)
    del swin, teacher
    for name, path in [("ElasticFace-Arc", WEIGHTS / "ElasticFace/295672backbone.pth"),
                       ("MaskInv-HG", WEIGHTS / "MaskInv/MaskInvHG_backbone.pth")]:
        m = load_r100(path)
        results[name] = run(m, m)
        del m
    torch.cuda.empty_cache()
    ins = load_model("r50", WEIGHTS / "InsightFace/16backbone.pth")
    results["InsightFace (R50)"] = run(ins, ins)
    for name in results:
        print(name, {n: round(results[name][n]["partial_vs_full"]["acc_mean"], 4) for n in benches}, flush=True)
    json.dump(results, open(OUT / out_name, "w"), indent=1)
    return results


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#dddcd7", "#fcfcfb"
    acc = {m: [100 * results[m][b]["partial_vs_full"]["acc_mean"] for b in BENCHES] for m in MODELS}
    lo = 5 * np.floor((min(min(v) for v in acc.values()) - 2) / 5)   # radial floor, rounded to 5 pts
    angles = np.linspace(0, 2 * np.pi, len(BENCHES), endpoint=False).tolist()

    fig = plt.figure(figsize=(12.5, 6.4), facecolor=surface)
    ax = fig.add_axes([0.05, 0.06, 0.48, 0.74], polar=True, facecolor=surface)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(lo, 100)
    ticks = np.arange(lo, 100.1, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.0f}%" for t in ticks], color=ink2, fontsize=8)
    ax.set_rlabel_position(90 / len(BENCHES))
    ax.set_xticks(angles)
    ax.set_xticklabels(BENCHES, color=ink, fontsize=11)
    ax.tick_params(axis="x", pad=10)
    ax.grid(color=grid, linewidth=0.8)
    ax.spines["polar"].set_color(grid)

    for m, style in MODELS.items():
        v = acc[m] + acc[m][:1]
        a = angles + angles[:1]
        ax.fill(a, v, color=style["color"], alpha=0.08, zorder=2)
        ax.plot(a, v, color=style["color"], linewidth=2, zorder=3, label=m)
        ax.scatter(angles, acc[m], s=36, color=style["color"], edgecolors=surface, linewidths=2, zorder=4)

    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.60, 0.88), ncol=2,
               frameon=False, fontsize=10, labelcolor=ink)
    fig.text(0.03, 0.95, f"Partial (ROI {RATIO:.2f}) vs full-face verification", fontsize=15, color=ink,
             weight="bold")
    fig.text(0.03, 0.915, "10-fold mean accuracy on each benchmark's official pairs; "
             f"radial axis starts at {lo:.0f}%", fontsize=10, color=ink2)

    # Exact values beside the chart (the radar is for shape; read numbers here).
    tab = fig.add_axes([0.60, 0.18, 0.38, 0.52])
    tab.axis("off")
    rows = [[f"{acc[m][i]:.2f}" for m in MODELS] for i in range(len(BENCHES))]
    rows.append([f"{np.mean(acc[m]):.2f}" for m in MODELS])
    t = tab.table(cellText=rows, rowLabels=BENCHES + ["Mean"],
                  colLabels=["Swin-B\n(ours)", "Elastic-\nFace-Arc", "MaskInv-\nHG", "Insight-\nFace"],
                  cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.9)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(grid)
        cell.set_facecolor(surface)
        cell.get_text().set_color(ink)
        if r == 0:
            cell.get_text().set_weight("bold")
            cell.set_height(cell.get_height() * 1.4)
        if r > 0 and c >= 0:   # bold the best model per benchmark
            vals = [float(x) for x in rows[r - 1]]
            if float(rows[r - 1][c]) == max(vals):
                cell.get_text().set_weight("bold")
    fig.text(0.60, 0.73, "Accuracy (%) — best per row in bold", fontsize=10, color=ink2)
    fig.text(0.60, 0.13, "Swin-B: ROI probe → Swin-B, full reference → its R50 teacher.\n"
             "Others: off-the-shelf; one model embeds both probe and reference.", fontsize=8.5, color=ink2)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"radar_partial_vs_full_roi15.{ext}", dpi=200, facecolor=surface)
    print(f"-> {OUT / 'radar_partial_vs_full_roi15.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="only re-plot from results.json")
    ap.add_argument("--suite", choices=["standard", "rfw"], default="standard")
    args = ap.parse_args()
    if args.suite == "rfw":
        evaluate([f"RFW-{r}" for r in RFW], "results_rfw.json")   # tables only, no radar
    else:
        plot(json.load(open(OUT / "results.json")) if args.plot else evaluate(BENCHES, "results.json"))
