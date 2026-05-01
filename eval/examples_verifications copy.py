import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# =========================
# Config
# =========================
QUERY_DIR   = "/datasets/glint360k/ROIs/example_verifacation/ratio_20/test"
GALLERY_DIR = "/datasets/glint360k/ROIs/example_verifacation/ratio_100/test"  
OUT_PATH    = "work_dirs/clip_ratio_20/paper_query_gallery_scores_improved.png"

NUM_IDS = 5
GALLERY_PER_ID = 2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG_PATH = "configs/variants_config_ratio_20.py"


# =========================
# Load both models
# =========================
def load_models():
    from backbones import get_model
    from utils.utils_config import get_config

    cfg = get_config(CONFIG_PATH)

    partial_face_backbone = get_model(
        cfg.network,
        dropout=0.0,
        fp16=cfg.fp16,
        num_features=cfg.embedding_size
    ).to(DEVICE)
    partial_face_backbone.load_state_dict(
        torch.load("work_dirs/clip_ratio_20/best_model.pt", map_location=DEVICE)
    )

    fullface_backbone = get_model(
        cfg.teacher_network,
        dropout=0.0,
        fp16=cfg.fp16,
        num_features=cfg.embedding_size
    ).to(DEVICE)
    fullface_backbone.load_state_dict(
        torch.load(cfg.teacher_model_path, map_location=DEVICE)
    )

    partial_face_backbone.eval()
    fullface_backbone.eval()
    return partial_face_backbone, fullface_backbone


# =========================
# Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

display_transform = transforms.Compose([
    transforms.Resize((112, 112)),
])


# =========================
# Utils
# =========================
@torch.no_grad()
def encode_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    emb = model(x)
    if isinstance(emb, (tuple, list)):
        emb = emb[0]

    emb = F.normalize(emb, dim=1)
    return emb[0].cpu()


def build_class_to_paths(root_dir):
    ds = datasets.ImageFolder(root=root_dir)
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

    class_to_paths = {}
    for path, class_idx in ds.samples:
        class_name = idx_to_class[class_idx]
        class_to_paths.setdefault(class_name, []).append(path)

    return class_to_paths


def sample_ids_and_images(query_dir, gallery_dir, num_ids=5, gallery_per_id=2, seed=42):
    random.seed(seed)

    query_map = build_class_to_paths(query_dir)
    gallery_map = build_class_to_paths(gallery_dir)

    common_ids = sorted(set(query_map.keys()) & set(gallery_map.keys()))
    valid_ids = [
        cid for cid in common_ids
        if len(query_map[cid]) >= 1 and len(gallery_map[cid]) >= gallery_per_id
    ]

    assert len(valid_ids) >= num_ids, f"Need at least {num_ids} valid IDs, found {len(valid_ids)}"

    selected_ids = random.sample(valid_ids, num_ids)

    query_paths = []
    gallery_by_id = {}

    for cid in selected_ids:
        qpath = random.choice(query_map[cid])
        gpaths = random.sample(gallery_map[cid], gallery_per_id)

        query_paths.append({"id": cid, "path": qpath})
        gallery_by_id[cid] = gpaths

    return selected_ids, query_paths, gallery_by_id


def add_colored_border(ax, color="green", lw=4):
    rect = Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=lw,
        clip_on=False
    )
    ax.add_patch(rect)


def make_ordered_gallery_for_query(query_id, selected_ids, gallery_by_id):
    ordered = []

    for p in gallery_by_id[query_id]:
        ordered.append({"id": query_id, "path": p, "tag": "same"})

    for other_id in selected_ids:
        if other_id == query_id:
            continue
        for p in gallery_by_id[other_id]:
            ordered.append({"id": other_id, "path": p, "tag": "diff"})

    return ordered


def make_figure(partial_model, full_model, selected_ids, query_paths, gallery_by_id, out_path):
    n_rows = len(query_paths)
    n_gallery_cols = sum(len(v) for v in gallery_by_id.values())
    n_cols = 1 + n_gallery_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.15 * n_cols, 2.9 * n_rows)
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    axes[0, 0].set_title("Query", fontsize=12)

    first_query_id = query_paths[0]["id"]
    first_gallery_items = make_ordered_gallery_for_query(first_query_id, selected_ids, gallery_by_id)
    for col, item in enumerate(first_gallery_items, start=1):
        # title = f"{item['tag'].upper()}\nID={item['id']}"
        title = f"{item['tag'].upper()}"
        axes[0, col].set_title(title, fontsize=10)

    for row, q in enumerate(query_paths):
        q_id = q["id"]
        q_path = q["path"]

        q_img = display_transform(Image.open(q_path).convert("RGB"))
        q_emb = encode_image(partial_model, q_path)

        gallery_items = make_ordered_gallery_for_query(q_id, selected_ids, gallery_by_id)

        gallery_embs = []
        for item in gallery_items:
            g_emb = encode_image(full_model, item["path"])
            gallery_embs.append(g_emb)
        gallery_embs = torch.stack(gallery_embs, dim=0)

        scores = torch.mv(gallery_embs, q_emb)

        ax = axes[row, 0]
        ax.imshow(q_img)
        ax.axis("off")
        add_colored_border(ax, color="black", lw=3)
        ax.set_ylabel(f"Query ID={q_id}", fontsize=11)

        for col, item in enumerate(gallery_items, start=1):
            g_id = item["id"]
            g_path = item["path"]
            tag = item["tag"]
            score = float(scores[col - 1].item())

            g_img = display_transform(Image.open(g_path).convert("RGB"))
            ax = axes[row, col]
            ax.imshow(g_img)
            ax.axis("off")

            border_color = "green" if tag == "same" else "red"
            add_colored_border(ax, color=border_color, lw=4)

            ax.text(
                0.5, -0.12,
                f"ID={g_id} | {score:.3f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color=border_color,
                fontweight="bold"
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved figure to: {out_path}")


def main():
    partial_model, full_model = load_models()

    selected_ids, query_paths, gallery_by_id = sample_ids_and_images(
        QUERY_DIR,
        GALLERY_DIR,
        num_ids=NUM_IDS,
        gallery_per_id=GALLERY_PER_ID,
        seed=SEED
    )

    print("Selected IDs:", selected_ids)

    make_figure(
        partial_model=partial_model,
        full_model=full_model,
        selected_ids=selected_ids,
        query_paths=query_paths,
        gallery_by_id=gallery_by_id,
        out_path=OUT_PATH
    )


if __name__ == "__main__":
    main()