"""t-SNE visualization of the "disjoint embedding space" problem and its fix:
teacher (full-face) vs student (partial-face) embeddings, BEFORE training
(fresh student init -> spaces are unaligned) and AFTER training (best_model
from clip_ratio_15 -> student learns to land in the teacher's space).
"""
import os
import sys
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from backbones import get_model
from dataset import crop_roi, get_transform

VAL_DIR = "/DATA/glint360k_arch_comparison/val"
VAL_METADATA_PATH = "/DATA/glint360k_arch_comparison/val_eye_center_metadata.json"
TEACHER_CKPT = "16backbone.pth"
STUDENT_CKPT = "work_dirs/exp_clip_r50_vs_swin_glint/clip_ratio_15/best_model.pt"
ROI_RATIO = 0.15
OUT_PATH = "work_dirs/exp_clip_r50_vs_swin_glint/embedding_tsne_analysis.png"

N_IDENTITIES = 15
N_IMAGES_PER_ID = 10
SEED = 7


def pick_identities(metadata, n_ids, n_per_id, seed):
    by_id = {}
    for k, v in metadata.items():
        if v.get("detected", False):
            by_id.setdefault(k.split("/", 1)[0], []).append(k)
    qualifying = [id_ for id_, keys in by_id.items() if len(keys) >= n_per_id]
    rng = random.Random(seed)
    rng.shuffle(qualifying)
    chosen = qualifying[:n_ids]
    selection = {}
    for id_ in chosen:
        keys = by_id[id_][:]
        rng.shuffle(keys)
        selection[id_] = keys[:n_per_id]
    return selection


@torch.no_grad()
def embed_batch(model, tensor, device="cuda", batch_size=64):
    model.eval()
    out = []
    for i in range(0, tensor.shape[0], batch_size):
        batch = tensor[i:i + batch_size].to(device)
        with torch.cuda.amp.autocast(True):
            emb = model(batch)
        out.append(F.normalize(emb.float()).cpu())
    return torch.cat(out).numpy()


def main():
    device = "cuda"
    metadata = json.load(open(VAL_METADATA_PATH))
    selection = pick_identities(metadata, N_IDENTITIES, N_IMAGES_PER_ID, SEED)
    print(f"Selected {len(selection)} identities x {N_IMAGES_PER_ID} images")

    transform = get_transform(augmentations=False)
    full_tensors, partial_tensors, labels = [], [], []
    id_to_idx = {id_: i for i, id_ in enumerate(selection)}
    for id_, keys in selection.items():
        for key in keys:
            filename = key.split("/", 1)[1]
            bgr = cv2.imread(os.path.join(VAL_DIR, id_, filename))
            if bgr is None:
                continue
            partial_bgr = crop_roi(bgr, ROI_RATIO, metadata[key]["center_y"], 1.0)
            full_tensors.append(transform(_to_pil(bgr)))
            partial_tensors.append(transform(_to_pil(partial_bgr)))
            labels.append(id_to_idx[id_])
    full_tensor = torch.stack(full_tensors)
    partial_tensor = torch.stack(partial_tensors)
    labels = np.array(labels)
    print(f"Total images embedded per model: {len(labels)}")

    teacher = get_model("r50", dropout=0.0, fp16=True, num_features=512).to(device)
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=device))
    teacher_full_emb = embed_batch(teacher, full_tensor, device)

    student_before = get_model("swin_base", dropout=0.0, fp16=True, num_features=512).to(device)
    student_before_emb = embed_batch(student_before, partial_tensor, device)
    del student_before

    student_after = get_model("swin_base", dropout=0.0, fp16=True, num_features=512).to(device)
    student_after.load_state_dict(torch.load(STUDENT_CKPT, map_location=device))
    student_after_emb = embed_batch(student_after, partial_tensor, device)
    del student_after

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cmap = plt.get_cmap("tab20")

    for ax, student_emb, title in [
        (axes[0], student_before_emb, "Before training\n(fresh student init)"),
        (axes[1], student_after_emb, "After training\n(clip_ratio_15 best_model)"),
    ]:
        combined = np.concatenate([teacher_full_emb, student_emb], axis=0)
        tsne = TSNE(n_components=2, perplexity=15, init="pca", random_state=SEED)
        proj = tsne.fit_transform(combined)
        n = len(labels)
        proj_teacher, proj_student = proj[:n], proj[n:]

        for id_idx in range(len(selection)):
            mask = labels == id_idx
            color = cmap(id_idx % 20)
            ax.scatter(proj_teacher[mask, 0], proj_teacher[mask, 1], color=color,
                       marker="o", s=45, edgecolors="black", linewidths=0.4,
                       label=f"id {id_idx}" if ax is axes[0] else None)
            ax.scatter(proj_student[mask, 0], proj_student[mask, 1], color=color,
                       marker="^", s=45, edgecolors="black", linewidths=0.4)

        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    # Domain legend (marker shape), shared
    from matplotlib.lines import Line2D
    domain_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=9, label="Teacher (full-face)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=9, label="Student (partial-face)"),
    ]
    fig.legend(handles=domain_handles, loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Embedding-space alignment: teacher (full-face) vs student (partial-face, ratio=0.15)\n"
        "Same color = same identity. Circles and triangles overlapping after training = disjoint-space problem solved.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160)
    plt.close(fig)
    print(f"Saved -> {OUT_PATH}")


def _to_pil(bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    main()
