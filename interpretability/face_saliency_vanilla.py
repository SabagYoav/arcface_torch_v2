import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms


# ============================================================
# 1. Utilities
# ============================================================

def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


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
    def __init__(self, image_size=(112, 112),
                 mean=(0.5, 0.5, 0.5),
                 std=(0.5, 0.5, 0.5)):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def load(self, image_path: str | Path) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img)  # [3,H,W]
        return tensor.unsqueeze(0)    # [1,3,H,W]


# ============================================================
# 3. Pair similarity saliency
# ============================================================

class FaceSimilarityExplainer:
    def __init__(self, model: torch.nn.Module, device="cuda"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.model(x)
        emb = F.normalize(emb, dim=1)
        return emb

    def cosine_score(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1, x2: [1,3,H,W]
        returns scalar cosine similarity
        """
        e1 = self.model(x1)
        e2 = self.model(x2)

        e1 = F.normalize(e1, dim=1)
        e2 = F.normalize(e2, dim=1)

        score = torch.sum(e1 * e2, dim=1)  # [1]
        return score[0]

    def saliency_for_pair(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mode: str = "similarity",
        abs_grads: bool = True,
    ):
        """
        mode:
            'similarity' -> explain what increases cosine similarity
            'distance'   -> explain what increases cosine distance = 1-cos
        returns:
            heat1, heat2, score_value
        """
        x1 = x1.clone().detach().to(self.device).requires_grad_(True)
        x2 = x2.clone().detach().to(self.device).requires_grad_(True)

        self.model.zero_grad(set_to_none=True)

        score = self.cosine_score(x1, x2)

        if mode == "similarity":
            target = score
        elif mode == "distance":
            target = 1.0 - score
        else:
            raise ValueError("mode must be 'similarity' or 'distance'")

        target.backward()

        g1 = x1.grad.detach()[0]  # [3,H,W]
        g2 = x2.grad.detach()[0]  # [3,H,W]

        if abs_grads:
            g1 = g1.abs()
            g2 = g2.abs()

        # channel aggregation
        heat1 = g1.max(dim=0).values.cpu().numpy()   # [H,W]
        heat2 = g2.max(dim=0).values.cpu().numpy()

        heat1 = normalize_map(heat1)
        heat2 = normalize_map(heat2)

        return heat1, heat2, score.item()

    def integrated_gradients_for_pair(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mode: str = "similarity",
        steps: int = 50,
        baseline_type: str = "zeros",
    ):
        """
        Integrated gradients for pair similarity.
        More stable than vanilla saliency.
        """
        x1 = x1.clone().detach().to(self.device)
        x2 = x2.clone().detach().to(self.device)

        if baseline_type == "zeros":
            b1 = torch.zeros_like(x1)
            b2 = torch.zeros_like(x2)
        else:
            raise ValueError("Only baseline_type='zeros' is implemented")

        total_grad_1 = torch.zeros_like(x1)
        total_grad_2 = torch.zeros_like(x2)

        for alpha in torch.linspace(0.0, 1.0, steps, device=self.device):
            xi1 = (b1 + alpha * (x1 - b1)).detach().requires_grad_(True)
            xi2 = (b2 + alpha * (x2 - b2)).detach().requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            score = self.cosine_score(xi1, xi2)

            if mode == "similarity":
                target = score
            elif mode == "distance":
                target = 1.0 - score
            else:
                raise ValueError("mode must be 'similarity' or 'distance'")

            target.backward()

            total_grad_1 += xi1.grad.detach()
            total_grad_2 += xi2.grad.detach()

        avg_grad_1 = total_grad_1 / steps
        avg_grad_2 = total_grad_2 / steps

        ig1 = (x1 - b1) * avg_grad_1
        ig2 = (x2 - b2) * avg_grad_2

        heat1 = ig1.abs()[0].max(dim=0).values.cpu().numpy()
        heat2 = ig2.abs()[0].max(dim=0).values.cpu().numpy()

        heat1 = normalize_map(heat1)
        heat2 = normalize_map(heat2)

        with torch.no_grad():
            score_value = self.cosine_score(x1, x2).item()

        return heat1, heat2, score_value


# ============================================================
# 4. Example model loader
# ============================================================

def load_arcface_backbone(model: torch.nn.Module, ckpt_path: str, device="cuda"):
    """
    Adapt this to your checkpoint format.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    # remove 'module.' prefix if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        new_state_dict[nk] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


# ============================================================
# 5. End-to-end runner
# ============================================================

def explain_face_pair(
    model: torch.nn.Module,
    img1_path: str,
    img2_path: str,
    out_dir: str,
    device: str = "cuda",
    method: str = "integrated_gradients",   # "saliency" or "integrated_gradients"
    mode: str = "similarity",               # "similarity" or "distance"
    image_size=(112, 112),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    pre = FacePreprocessor(image_size=image_size, mean=mean, std=std)
    explainer = FaceSimilarityExplainer(model, device=device)

    x1 = pre.load(img1_path)
    x2 = pre.load(img2_path)

    if method == "saliency":
        heat1, heat2, score = explainer.saliency_for_pair(x1, x2, mode=mode)
    elif method == "integrated_gradients":
        heat1, heat2, score = explainer.integrated_gradients_for_pair(x1, x2, mode=mode, steps=50)
    else:
        raise ValueError("method must be 'saliency' or 'integrated_gradients'")

    rgb1 = tensor_to_rgb_image(x1, mean=mean, std=std)
    rgb2 = tensor_to_rgb_image(x2, mean=mean, std=std)

    overlay1 = overlay_heatmap_on_image(rgb1, heat1)
    overlay2 = overlay_heatmap_on_image(rgb2, heat2)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(out_dir / "img1_overlay.jpg", overlay1)
    save_rgb(out_dir / "img2_overlay.jpg", overlay2)

    save_rgb(out_dir / "img1_original.jpg", rgb1)
    save_rgb(out_dir / "img2_original.jpg", rgb2)

    cv2.imwrite(str(out_dir / "img1_heatmap.png"), (heat1 * 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "img2_heatmap.png"), (heat2 * 255).astype(np.uint8))

    concat = np.concatenate([overlay1, overlay2], axis=1)
    save_rgb(out_dir / "pair_overlay.jpg", concat)

    print(f"Cosine score: {score:.6f}")
    print(f"Saved results to: {out_dir}")


# ============================================================
# 6. Example usage
# ============================================================

from backbones import get_model

if __name__ == "__main__":
    # --------------------------------------------------------
    # Replace this with your own backbone constructor
    # Example:
    # from backbones import get_model
    model = get_model(
        'r50', dropout=0.0, fp16=False, num_features=512).cuda()
    checkpoint_fpath='work_dirs/exp_glint360k_roi_100_r50_arcface/best_model.pt'
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint)
    # --------------------------------------------------------
    explain_face_pair(
    model=model,
    img1_path="/datasets/glint360k/imageFolder_split_fullface/test/4/263.jpg",
    img2_path="/datasets/glint360k/imageFolder_split_fullface/test/4/280.jpg",
    out_dir="outputs/face_saliency",
    device="cuda",
    method="integrated_gradients",
    mode="similarity",
    image_size=(112, 112),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)
