import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# Config
# =========================
RATIO = 20
CONFIG_PATH = f"configs/variants_config_ratio_{RATIO}.py"
PARTIAL_DIR = f"/datasets/glint360k/ROIs/ratio_{RATIO}/test"
FULL_DIR = "/datasets/glint360k/ROIs/ratio_100/test"
OUT_ROC_IMG = f"work_dirs/clip_ratio_{RATIO}/roc_partial_vs_full.png"

BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Load models for a given ratio
# =========================
def load_models():
    from backbones import get_model
    from utils.utils_config import get_config
    cfg = get_config(CONFIG_PATH)
    partial_face_backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(DEVICE)
    partial_face_backbone.load_state_dict(torch.load(f"{cfg.output}/best_model.pt", map_location=DEVICE, weights_only=True))

    fullface_backbone = get_model(cfg.teacher_network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(DEVICE)
    fullface_backbone.load_state_dict(torch.load(cfg.teacher_model_path, map_location=DEVICE, weights_only=True))

    partial_face_backbone.eval()
    fullface_backbone.eval()
    return partial_face_backbone, fullface_backbone


# =========================
# Image transform
# =========================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# =========================
# Extract embeddings
# =========================
@torch.no_grad()
def extract_embeddings(model, root_dir):
    ds = datasets.ImageFolder(root=root_dir, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    all_embs = []
    all_labels = []

    for imgs, labels in tqdm(dl, desc=f"Extracting {root_dir}"):
        imgs = imgs.to(DEVICE, non_blocking=True)

        emb = model(imgs)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]

        emb = F.normalize(emb, dim=1)

        all_embs.append(emb.cpu())
        all_labels.append(labels.cpu())

        #TODO: remove this break
        if len(all_embs) >= 200:  # Just process 100 batches for quick testing
            break

    all_embs = torch.cat(all_embs, dim=0)    # [N, D]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    return all_embs, all_labels


# =========================
# Main
# =========================
def main():
    partial_face_model, fullface_model = load_models()

    partial_embs, partial_labels = extract_embeddings(partial_face_model, PARTIAL_DIR)
    full_embs, full_labels = extract_embeddings(fullface_model, FULL_DIR)

    # cosine similarity matrix
    scores_mat = partial_embs @ full_embs.T   # [N_partial, N_full]

    # positive if same identity, else negative
    label_mat = (partial_labels[:, None] == full_labels[None, :]).int()

    scores = scores_mat.numpy().reshape(-1)
    labels = label_mat.numpy().reshape(-1)

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    print(f"AUC = {roc_auc:.6f}")

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC: Partial Face ({RATIO}%) vs Full Face")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_ROC_IMG, dpi=200)
    plt.close()

    print(f"ROC curve saved to: {OUT_ROC_IMG}")


if __name__ == "__main__":
    main()