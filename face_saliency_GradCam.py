import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from backbones import get_model


# ============================================================
# utils
# ============================================================

def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def tensor_to_rgb_image(
    tensor: torch.Tensor,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_heatmap_on_image(
    rgb_img: np.ndarray,
    heatmap: np.ndarray,
    alpha_img: float = 0.65,
    alpha_heatmap: float = 0.45,
) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (
        alpha_img * rgb_img.astype(np.float32)
        + alpha_heatmap * heatmap_color.astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_rgb(path: str | Path, rgb: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


# ============================================================
# preprocessing
# ============================================================

class FacePreprocessor:
    def __init__(
        self,
        image_size=(112, 112),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def load_fullface(self, image_path: str | Path) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0)

    def load_partialface(self, image_path: str | Path) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.transform(img).unsqueeze(0)


# ============================================================
# model loading
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
# grad-cam helpers
# ============================================================

def find_last_conv_layer(model: nn.Module):
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module
    return last_name, last_module


class SingleModelGradCAM:
    """
    Handles Grad-CAM for one model and one hooked target layer.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module, device="cuda"):
        self.model = model.to(device).eval()
        self.device = device
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def _make_cam(self):
        acts = self.activations[0]   # [C,H,W]
        grads = self.gradients[0]    # [C,H,W]

        weights = grads.mean(dim=(1, 2), keepdim=True)   # [C,1,1]
        cam = torch.sum(weights * acts, dim=0)           # [H,W]
        # cam = torch.abs(weights * acts)           # [H,W]
        # cam = torch.sum(torch.abs(weights * acts), dim=0) 
        cam = F.relu(cam)
        # cam = sum(cam)  # in case of multiple channels, sum them up (or take max, etc.)

        cam = cam.detach().cpu().numpy()
        cam = normalize_map(cam)
        return cam


# ============================================================
# dual-encoder pair grad-cam
# ============================================================

class DualEncoderPairGradCAM:
    def __init__(
        self,
        fullface_model: nn.Module,
        partialface_model: nn.Module,
        fullface_target_layer: nn.Module,
        partialface_target_layer: nn.Module,
        device="cuda",
    ):
        self.device = device

        self.full_cam = SingleModelGradCAM(
            model=fullface_model,
            target_layer=fullface_target_layer,
            device=device,
        )

        self.partial_cam = SingleModelGradCAM(
            model=partialface_model,
            target_layer=partialface_target_layer,
            device=device,
        )

    def remove(self):
        self.full_cam.remove()
        self.partial_cam.remove()

    def explain_fullface(
        self,
        fullface_x: torch.Tensor,
        partialface_x: torch.Tensor,
    ):
        """
        CAM for full-face image, with partial embedding frozen as reference.
        """
        fullface_x = fullface_x.to(self.device)
        partialface_x = partialface_x.to(self.device)

        with torch.no_grad():
            part_ref = F.normalize(self.partial_cam.model(partialface_x), dim=1)

        self.full_cam.model.zero_grad(set_to_none=True)
        emb_full = F.normalize(self.full_cam.model(fullface_x), dim=1)

        score = torch.sum(emb_full * part_ref, dim=1)[0]
        score.backward()

        cam = self.full_cam._make_cam()
        return cam, score.item()

    def explain_partialface(
        self,
        fullface_x: torch.Tensor,
        partialface_x: torch.Tensor,
    ):
        """
        CAM for partial-face image, with full embedding frozen as reference.
        """
        fullface_x = fullface_x.to(self.device)
        partialface_x = partialface_x.to(self.device)

        with torch.no_grad():
            full_ref = F.normalize(self.full_cam.model(fullface_x), dim=1)

        self.partial_cam.model.zero_grad(set_to_none=True)
        emb_part = F.normalize(self.partial_cam.model(partialface_x), dim=1)

        score = torch.sum(full_ref * emb_part, dim=1)[0]
        score.backward()

        cam = self.partial_cam._make_cam()
        return cam, score.item()


# ============================================================
# end-to-end runner
# ============================================================

def run_dual_encoder_pair_gradcam(
    fullface_ckpt_path: str,
    partialface_ckpt_path: str,
    fullface_img_path: str,
    partialface_img_path: str,
    out_dir: str = "outputs/dual_encoder_pair_gradcam",
    network_full: str = "r50",
    network_partial: str = "r50",
    device: str = "cuda",
    image_size=(112, 112),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
):
    fullface_model = load_arcface_backbone(
        ckpt_path=fullface_ckpt_path,
        device=device,
        network=network_full,
    )
    partialface_model = load_arcface_backbone(
        ckpt_path=partialface_ckpt_path,
        device=device,
        network=network_partial,
    )

    ## Find the last conv layer in each model for Grad-CAM
    # full_layer_name, full_target_layer = find_last_conv_layer(fullface_model)
    partial_layer_name, partial_target_layer = find_last_conv_layer(partialface_model)
    partial_layer_name, partial_target_layer = find_last_conv_layer(partialface_model)

    full_named = dict(fullface_model.named_modules())
    # partial_named = dict(partialface_model.named_modules())
    
    full_target_layer = full_named["layer3.0.conv2"]      # try several options here
    # partial_target_layer = partial_named["layer4.2.conv3"] 


    pre = FacePreprocessor(
        image_size=image_size,
        mean=mean,
        std=std,
    )

    fullface_x = pre.load_fullface(fullface_img_path)
    partialface_x = pre.load_partialface(partialface_img_path)

    explainer = DualEncoderPairGradCAM(
        fullface_model=fullface_model,
        partialface_model=partialface_model,
        fullface_target_layer=full_target_layer,
        partialface_target_layer=partial_target_layer,
        device=device,
    )

    cam_full, score_full = explainer.explain_fullface(fullface_x, partialface_x)
    cam_part, score_part = explainer.explain_partialface(fullface_x, partialface_x)

    explainer.remove()

    rgb_full = tensor_to_rgb_image(fullface_x, mean=mean, std=std)
    rgb_part = tensor_to_rgb_image(partialface_x, mean=mean, std=std)

    overlay_full = overlay_heatmap_on_image(rgb_full, cam_full)
    overlay_part = overlay_heatmap_on_image(rgb_part, cam_part)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_rgb(out_dir / "fullface_original.jpg", rgb_full)
    save_rgb(out_dir / "partialface_original.jpg", rgb_part)
    save_rgb(out_dir / "fullface_gradcam.jpg", overlay_full)
    save_rgb(out_dir / "partialface_gradcam.jpg", overlay_part)

    cv2.imwrite(
        str(out_dir / "fullface_cam.png"),
        (cv2.resize(cam_full, image_size) * 255).astype(np.uint8),
    )
    cv2.imwrite(
        str(out_dir / "partialface_cam.png"),
        (cv2.resize(cam_part, image_size) * 255).astype(np.uint8),
    )

    concat = np.concatenate([overlay_full, overlay_part], axis=1)
    save_rgb(out_dir / "pair_gradcam.jpg", concat)

    # true forward score with both models
    with torch.no_grad():
        full_emb = F.normalize(fullface_model(fullface_x.to(device)), dim=1)
        part_emb = F.normalize(partialface_model(partialface_x.to(device)), dim=1)
        cosine_score = torch.sum(full_emb * part_emb, dim=1)[0].item()

    print(f"Full explanation score:    {score_full:.6f}")
    print(f"Partial explanation score: {score_part:.6f}")
    print(f"Cosine similarity score:   {cosine_score:.6f}")
    print(f"Saved to: {out_dir}")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    run_dual_encoder_pair_gradcam(
        fullface_ckpt_path="work_dirs/exp_glint360k_roi_100_r50_arcface/best_model.pt",
        partialface_ckpt_path="work_dirs/clip_ratio_20/best_model.pt",
        fullface_img_path="/datasets/glint360k/ROIs/ratio_100/test/42/3010.jpg",
        partialface_img_path="/datasets/glint360k/ROIs/ratio_20/test/84/5153.jpg",
        out_dir="outputs/dual_encoder_pair_gradcam",
        network_full="r50",
        network_partial="vit_s_dp005_mask_0",
        device="cuda",
        image_size=(112, 112),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )