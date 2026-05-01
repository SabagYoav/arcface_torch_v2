import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from backbones import get_model


# ============================================================
# 1. Utilities
# ============================================================

def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

def emphasize_heatmap(
    heatmap: np.ndarray,
    blur_ksize: int = 9,
    gamma: float = 0.65,
    contrast: float = 2.2,
):
    """
    Emphasize salient areas without removing weaker ones.
    Keeps all regions, but boosts stronger responses.
    """
    h = heatmap.astype(np.float32).copy()

    if blur_ksize > 1:
        h = cv2.GaussianBlur(h, (blur_ksize, blur_ksize), 0)

    # normalize
    h = h - h.min()
    if h.max() > 0:
        h = h / h.max()

    # # boost contrast around mid/high values
    h = np.clip((h - 0.5) * contrast + 0.5, 0, 1)

    # gamma < 1 emphasizes brighter responses but keeps all values
    h = np.power(h, gamma)

    return np.clip(h, 0, 1)

def overlay_heatmap_on_image_strong(
    rgb_img: np.ndarray,
    heatmap: np.ndarray,
    alpha_img: float = 0.55,
    alpha_heatmap: float = 0.95,
):
    """
    Stronger overlay without deleting heatmap regions.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (
        alpha_img * rgb_img.astype(np.float32)
        + alpha_heatmap * heatmap_color.astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def tensor_to_rgb_image(
    tensor: torch.Tensor,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> np.ndarray:
    """
    tensor: [1,3,H,W] or [3,H,W]
    returns uint8 RGB image [H,W,3]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def overlay_heatmap_on_image(rgb_img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    rgb_img: uint8 RGB [H,W,3]
    heatmap: float [H,W] in [0,1]
    returns RGB overlay
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = 0.5 * rgb_img.astype(np.float32) + 0.5 * heatmap_color.astype(np.float32)
    # alpha_img = 0.55
    # alpha_heatmap = 0.95
    # overlay = (
    #     alpha_img * rgb_img.astype(np.float32)
    #     + alpha_heatmap * heatmap_color.astype(np.float32)
    # )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def save_rgb(path: str | Path, rgb: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


# ============================================================
# 2. Preprocessing
# ============================================================

class FacePreprocessor:
    def __init__(
        self,
        image_size=(112, 112),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def load_fullface(self, image_path: str | Path) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img)
        return tensor.unsqueeze(0)

    def load_partialface(self, image_path: str | Path) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img)
        return tensor.unsqueeze(0)


# ============================================================
# 3. Dual-encoder pair similarity saliency
# ============================================================

class DualEncoderFaceSimilarityExplainer:
    def __init__(
        self,
        fullface_model: torch.nn.Module,
        partialface_model: torch.nn.Module,
        device="cuda",
    ):
        self.fullface_model = fullface_model.to(device).eval()
        self.partialface_model = partialface_model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed_fullface(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.fullface_model(x.to(self.device))
        emb = F.normalize(emb, dim=1)
        return emb

    @torch.no_grad()
    def embed_partialface(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.partialface_model(x.to(self.device))
        emb = F.normalize(emb, dim=1)
        return emb

    def cosine_score(self, fullface_x: torch.Tensor, partialface_x: torch.Tensor) -> torch.Tensor:
        """
        fullface_x:   [1,3,H,W]
        partialface_x:[1,3,H,W]
        returns scalar cosine similarity
        """
        full_emb = self.fullface_model(fullface_x)
        part_emb = self.partialface_model(partialface_x)

        full_emb = F.normalize(full_emb, dim=1)
        part_emb = F.normalize(part_emb, dim=1)

        score = torch.sum(full_emb * part_emb, dim=1)  # [1]
        return score[0]

    def saliency_for_pair(
        self,
        fullface_x: torch.Tensor,
        partialface_x: torch.Tensor,
        mode: str = "similarity",
        abs_grads: bool = True,
    ):
        fullface_x = fullface_x.clone().detach().to(self.device).requires_grad_(True)
        partialface_x = partialface_x.clone().detach().to(self.device).requires_grad_(True)

        self.fullface_model.zero_grad(set_to_none=True)
        self.partialface_model.zero_grad(set_to_none=True)

        score = self.cosine_score(fullface_x, partialface_x)

        if mode == "similarity":
            target = score
        elif mode == "distance":
            target = 1.0 - score
        else:
            raise ValueError("mode must be 'similarity' or 'distance'")

        target.backward()

        g_full = fullface_x.grad.detach()[0]      # [3,H,W]
        g_part = partialface_x.grad.detach()[0]   # [3,H,W]

        if abs_grads:
            g_full = g_full.abs()
            g_part = g_part.abs()

        heat_full = g_full.max(dim=0).values.cpu().numpy()
        heat_part = g_part.max(dim=0).values.cpu().numpy()

        heat_full = normalize_map(heat_full)
        heat_part = normalize_map(heat_part)

        return heat_full, heat_part, score.item()

    def integrated_gradients_for_pair(
        self,
        fullface_x: torch.Tensor,
        partialface_x: torch.Tensor,
        mode: str = "similarity",
        steps: int = 50,
        baseline_type: str = "zeros",
    ):
        fullface_x = fullface_x.clone().detach().to(self.device)
        partialface_x = partialface_x.clone().detach().to(self.device)

        if baseline_type == "zeros":
            full_base = torch.zeros_like(fullface_x)
            part_base = torch.zeros_like(partialface_x)
        else:
            raise ValueError("Only baseline_type='zeros' is implemented")

        total_grad_full = torch.zeros_like(fullface_x)
        total_grad_part = torch.zeros_like(partialface_x)

        for alpha in torch.linspace(0.0, 1.0, steps, device=self.device):
            xi_full = (full_base + alpha * (fullface_x - full_base)).detach().requires_grad_(True)
            xi_part = (part_base + alpha * (partialface_x - part_base)).detach().requires_grad_(True)

            self.fullface_model.zero_grad(set_to_none=True)
            self.partialface_model.zero_grad(set_to_none=True)

            score = self.cosine_score(xi_full, xi_part)

            if mode == "similarity":
                target = score
            elif mode == "distance":
                target = 1.0 - score
            else:
                raise ValueError("mode must be 'similarity' or 'distance'")

            target.backward()

            total_grad_full += xi_full.grad.detach()
            total_grad_part += xi_part.grad.detach()

        avg_grad_full = total_grad_full / steps
        avg_grad_part = total_grad_part / steps

        ig_full = (fullface_x - full_base) * avg_grad_full
        ig_part = (partialface_x - part_base) * avg_grad_part

        heat_full = ig_full.abs()[0].max(dim=0).values.cpu().numpy()
        heat_part = ig_part.abs()[0].max(dim=0).values.cpu().numpy()

        heat_full = normalize_map(heat_full)
        heat_part = normalize_map(heat_part)

        with torch.no_grad():
            score_value = self.cosine_score(fullface_x, partialface_x).item()

        return heat_full, heat_part, score_value


# ============================================================
# 4. Model loader
# ============================================================

def load_arcface_backbone(
    ckpt_path: str,
    device: str = "cuda",
    network: str = "r50",
):
    model = get_model(network, dropout=0.0, fp16=False, num_features=512).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    return model


# ============================================================
# 5. End-to-end runner
# ============================================================

def explain_face_pair_dual_encoder(
    fullface_model: torch.nn.Module,
    partialface_model: torch.nn.Module,
    fullface_img1_path: str,
    partialface_img2_path: str,
    out_dir: str,
    device: str = "cuda",
    method: str = "saliency",   # "saliency" or "integrated_gradients"
    mode: str = "similarity",               # "similarity" or "distance"
    image_size=(112, 112),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    pre = FacePreprocessor(image_size=image_size, mean=mean, std=std)
    explainer = DualEncoderFaceSimilarityExplainer(
        fullface_model=fullface_model,
        partialface_model=partialface_model,
        device=device,
    )

    full_x = pre.load_fullface(fullface_img1_path)
    part_x = pre.load_partialface(partialface_img2_path)

    if method == "saliency":
        heat_full, heat_part, score = explainer.saliency_for_pair(
            full_x, part_x, mode=mode
        )
    elif method == "integrated_gradients":
        heat_full, heat_part, score = explainer.integrated_gradients_for_pair(
            full_x, part_x, mode=mode, steps=50
        )
    else:
        raise ValueError("method must be 'saliency' or 'integrated_gradients'")

    # heat_full = emphasize_heatmap(heat_full)
    # heat_part = emphasize_heatmap(heat_part)

    rgb_full = tensor_to_rgb_image(full_x, mean=mean, std=std)
    rgb_part = tensor_to_rgb_image(part_x, mean=mean, std=std)

    overlay_full = overlay_heatmap_on_image(rgb_full, heat_full)
    overlay_part = overlay_heatmap_on_image(rgb_part, heat_part)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(out_dir / "fullface_original.jpg", rgb_full)
    save_rgb(out_dir / "partialface_original.jpg", rgb_part)

    save_rgb(out_dir / "fullface_overlay.jpg", overlay_full)
    save_rgb(out_dir / "partialface_overlay.jpg", overlay_part)

    cv2.imwrite(str(out_dir / "fullface_heatmap.png"), (heat_full * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "partialface_heatmap.png"), (heat_part * 255).astype(np.uint8))

    concat = np.concatenate([overlay_full, overlay_part], axis=1)
    save_rgb(out_dir / "pair_overlay.jpg", concat)

    print(f"Cosine score: {score:.6f}")
    print(f"Saved results to: {out_dir}")


# ============================================================
# 6. Example usage
# ============================================================

if __name__ == "__main__":
    fullface_model = load_arcface_backbone(
        ckpt_path="work_dirs/exp_glint360k_roi_100_r50_arcface/best_model.pt",
        device="cuda",
        network="r50",
    )

    partialface_model = load_arcface_backbone(
        ckpt_path="work_dirs/clip_ratio_20/best_model.pt",
        device="cuda",
        network="vit_s_dp005_mask_0",
    )

    explain_face_pair_dual_encoder(
        fullface_model=fullface_model,
        partialface_model=partialface_model,
        # fullface_img1_path="/datasets/glint360k/imageFolder_split_fullface/test/4/263.jpg",
        fullface_img1_path="/datasets/glint360k/ROIs/ratio_100/test/42/3010.jpg",
        # partialface_img2_path="/datasets/glint360k/ROIs/ratio_20/test/4/322.jpg",
        partialface_img2_path="/datasets/glint360k/ROIs/ratio_20/test/42/3001.jpg",
        out_dir="outputs/face_saliency_dual_encoder",
        device="cuda",
        method="integrated_gradients",   # or "saliency"
        mode="similarity",
        image_size=(112, 112),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )