import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from backbones import get_model


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    mx = x.max()
    if mx > 1e-8:
        x = x / mx
    return x


def save_rgb(path: str | Path, rgb: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def tensor_to_rgb_image(
    x: torch.Tensor,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
) -> np.ndarray:
    if x.dim() == 4:
        x = x[0]

    x = x.detach().cpu().float().clone()
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    x = x * std_t + mean_t
    x = x.clamp(0, 1)

    rgb = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return rgb


def overlay_heatmap_on_image(
    rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_uint8 = (normalize_map(cam_resized) * 255).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    out = cv2.addWeighted(rgb, 1.0 - alpha, heat_rgb, alpha, 0)
    return out


def resize_rgb(rgb: np.ndarray, size=(224, 224)) -> np.ndarray:
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def add_text_bar(
    rgb: np.ndarray,
    text: str,
    bar_h: int = 24,
    font_scale: float = 0.48,
    thickness: int = 1,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[:h] = rgb
    cv2.putText(
        out,
        text,
        (4, h + bar_h - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def make_header_cell(
    text: str,
    size=(224, 248),
    font_scale=0.8,
    thickness=2,
) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x = max(8, (w - text_size[0]) // 2)
    y = (h + text_size[1]) // 2
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return img


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

    def _load(self, img_path: str | Path) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x.unsqueeze(0)

    def load_fullface(self, img_path: str | Path) -> torch.Tensor:
        return self._load(img_path)

    def load_partialface(self, img_path: str | Path) -> torch.Tensor:
        return self._load(img_path)


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


def find_last_conv_layer(model: nn.Module):
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
            last_module = module
    return last_name, last_module


def choose_target_layer(model: nn.Module, preferred_names: list[str] | None = None):
    named = dict(model.named_modules())

    if preferred_names is None:
        preferred_names = [
            "layer4.2.conv3",
            "layer4.1.conv3",
            "layer4.0.conv3",
            "layer3.5.conv3",
            "layer3.2.conv3",
            "layer3.1.conv2",
            "layer3.0.conv2",
            "layer4.2.conv2",
            "layer4.1.conv2",
            "layer4.0.conv2",
        ]

    for name in preferred_names:
        if name in named:
            return name, named[name]

    last_name, last_module = find_last_conv_layer(model)
    if last_module is None:
        raise ValueError("No Conv2d layer found for target layer.")
    return last_name, last_module


class SingleModelGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device="cuda"):
        self.model = model.to(device).eval()
        self.target_layer = target_layer
        self.device = device

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

    def make_cam(self):
        acts = self.activations[0]
        grads = self.gradients[0]

        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * acts, dim=0)
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        return normalize_map(cam)


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
        self.full_cam = SingleModelGradCAM(fullface_model, fullface_target_layer, device=device)
        self.partial_cam = SingleModelGradCAM(partialface_model, partialface_target_layer, device=device)

    def remove(self):
        self.full_cam.remove()
        self.partial_cam.remove()

    def explain_fullface(self, fullface_x: torch.Tensor, partialface_x: torch.Tensor):
        fullface_x = fullface_x.to(self.device)
        partialface_x = partialface_x.to(self.device)

        with torch.no_grad():
            part_ref = F.normalize(self.partial_cam.model(partialface_x), dim=1)

        self.full_cam.model.zero_grad(set_to_none=True)
        emb_full = F.normalize(self.full_cam.model(fullface_x), dim=1)

        score = torch.sum(emb_full * part_ref, dim=1)[0]
        score.backward()

        cam = self.full_cam.make_cam()
        return cam, score.item()

    def explain_partialface(self, fullface_x: torch.Tensor, partialface_x: torch.Tensor):
        fullface_x = fullface_x.to(self.device)
        partialface_x = partialface_x.to(self.device)

        with torch.no_grad():
            full_ref = F.normalize(self.full_cam.model(fullface_x), dim=1)

        self.partial_cam.model.zero_grad(set_to_none=True)
        emb_part = F.normalize(self.partial_cam.model(partialface_x), dim=1)

        score = torch.sum(full_ref * emb_part, dim=1)[0]
        score.backward()

        cam = self.partial_cam.make_cam()
        return cam, score.item()


def list_id_dirs(base_dir: str | Path) -> list[Path]:
    base_dir = Path(base_dir)
    return sorted([p for p in base_dir.iterdir() if p.is_dir()])


def list_images_in_id_dir(id_dir: str | Path) -> list[Path]:
    id_dir = Path(id_dir)
    return sorted([p for p in id_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def build_id_to_images(base_dir: str | Path) -> dict[str, list[Path]]:
    id_to_images = {}
    for id_dir in list_id_dirs(base_dir):
        imgs = list_images_in_id_dir(id_dir)
        if len(imgs) > 0:
            id_to_images[id_dir.name] = imgs
    return id_to_images


def sample_crossdir_query_rows(
    fullface_id_to_images: dict[str, list[Path]],
    partial_id_to_images: dict[str, list[Path]],
    num_rows: int = 5,
    seed: int = 42,
):
    rng = random.Random(seed)

    common_ids = sorted(set(fullface_id_to_images.keys()) & set(partial_id_to_images.keys()))
    valid_query_ids = [
        id_name for id_name in common_ids
        if len(fullface_id_to_images[id_name]) >= 1 and len(partial_id_to_images[id_name]) >= 2
    ]

    if len(valid_query_ids) < num_rows:
        raise ValueError(
            f"Need at least {num_rows} shared IDs with 1+ fullface and 2+ partial images."
        )

    partial_all_ids = list(partial_id_to_images.keys())
    if len(partial_all_ids) < 3:
        raise ValueError("Need at least 3 IDs in partial gallery.")

    chosen_query_ids = rng.sample(valid_query_ids, num_rows)

    rows = []
    for qid in chosen_query_ids:
        query_img = rng.choice(fullface_id_to_images[qid])
        same_imgs = rng.sample(partial_id_to_images[qid], 2)

        diff_ids = [x for x in partial_all_ids if x != qid]
        picked_diff_ids = rng.sample(diff_ids, 2)
        diff_imgs = [
            rng.choice(partial_id_to_images[picked_diff_ids[0]]),
            rng.choice(partial_id_to_images[picked_diff_ids[1]]),
        ]

        rows.append({
            "query_id": qid,
            "query_img": query_img,
            "same_imgs": same_imgs,
            "diff_imgs": diff_imgs,
        })

    return rows


def make_query_cam_cell(
    explainer: DualEncoderPairGradCAM,
    pre: FacePreprocessor,
    query_img_path: str | Path,
    reference_partial_img_path: str | Path,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    cell_size=(224, 224),
    label: str = "",
) -> np.ndarray:
    fullface_x = pre.load_fullface(query_img_path)
    partialface_x = pre.load_partialface(reference_partial_img_path)

    cam, sim = explainer.explain_fullface(fullface_x, partialface_x)

    rgb_full = tensor_to_rgb_image(fullface_x, mean=mean, std=std)
    overlay = overlay_heatmap_on_image(rgb_full, cam)
    overlay = resize_rgb(overlay, cell_size)

    return add_text_bar(overlay, f"{label} | sim={sim:.3f}")


def make_gallery_partial_cam_cell(
    explainer: DualEncoderPairGradCAM,
    pre: FacePreprocessor,
    query_img_path: str | Path,
    partial_img_path: str | Path,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    cell_size=(224, 224),
    label: str = "",
) -> np.ndarray:
    fullface_x = pre.load_fullface(query_img_path)
    partialface_x = pre.load_partialface(partial_img_path)

    cam_part, sim = explainer.explain_partialface(fullface_x, partialface_x)

    rgb_part = tensor_to_rgb_image(partialface_x, mean=mean, std=std)
    overlay = overlay_heatmap_on_image(rgb_part, cam_part)
    overlay = resize_rgb(overlay, cell_size)

    return add_text_bar(overlay, f"{label} | sim={sim:.3f}")


def stack_grid(rows_rgb: list[list[np.ndarray]]) -> np.ndarray:
    return np.concatenate([np.concatenate(row, axis=1) for row in rows_rgb], axis=0)


def run_dual_encoder_grid_gradcam(
    fullface_ckpt_path: str,
    partialface_ckpt_path: str,
    fullface_base_dir: str,
    partialface_base_dir: str,
    out_path: str = "outputs/dual_encoder_pair_gradcam/grid_5x5.jpg",
    network_full: str = "r50",
    network_partial: str = "r50",
    device: str = "cuda",
    image_size=(112, 112),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    seed: int = 42,
    cell_size=(224, 224),
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

    full_layer_name, full_target_layer = choose_target_layer(
        fullface_model,
        preferred_names=[
            "layer3.0.conv2",
            "layer3.1.conv2",
            "layer3.2.conv2",
            "layer4.0.conv2",
            "layer4.1.conv2",
            "layer4.2.conv2",
            "layer4.2.conv3",
        ],
    )
    partial_layer_name, partial_target_layer = choose_target_layer(partialface_model)

    print(f"[INFO] full target layer:    {full_layer_name}")
    print(f"[INFO] partial target layer: {partial_layer_name}")

    pre = FacePreprocessor(
        image_size=image_size,
        mean=mean,
        std=std,
    )

    explainer = DualEncoderPairGradCAM(
        fullface_model=fullface_model,
        partialface_model=partialface_model,
        fullface_target_layer=full_target_layer,
        partialface_target_layer=partial_target_layer,
        device=device,
    )

    fullface_id_to_images = build_id_to_images(fullface_base_dir)
    partial_id_to_images = build_id_to_images(partialface_base_dir)

    rows_meta = sample_crossdir_query_rows(
        fullface_id_to_images=fullface_id_to_images,
        partial_id_to_images=partial_id_to_images,
        num_rows=5,
        seed=seed,
    )

    header_row = [
        make_header_cell("query", size=(cell_size[0], cell_size[1] + 24)),
        make_header_cell("same id", size=(cell_size[0], cell_size[1] + 24)),
        make_header_cell("same id", size=(cell_size[0], cell_size[1] + 24)),
        make_header_cell("different id", size=(cell_size[0], cell_size[1] + 24)),
        make_header_cell("different id", size=(cell_size[0], cell_size[1] + 24)),
    ]

    grid_rows = [header_row]

    for row in rows_meta:
        qid = row["query_id"]
        query_img = row["query_img"]
        same1, same2 = row["same_imgs"]
        diff1, diff2 = row["diff_imgs"]

        query_reference_partial = same1

        row_cells = []

        row_cells.append(
            make_query_cam_cell(
                explainer=explainer,
                pre=pre,
                query_img_path=query_img,
                reference_partial_img_path=query_reference_partial,
                mean=mean,
                std=std,
                cell_size=cell_size,
                label=f"Q id={qid}",
            )
        )

        row_cells.append(
            make_gallery_partial_cam_cell(
                explainer=explainer,
                pre=pre,
                query_img_path=query_img,
                partial_img_path=same1,
                mean=mean,
                std=std,
                cell_size=cell_size,
                label=f"same {same1.parent.name}",
            )
        )

        row_cells.append(
            make_gallery_partial_cam_cell(
                explainer=explainer,
                pre=pre,
                query_img_path=query_img,
                partial_img_path=same2,
                mean=mean,
                std=std,
                cell_size=cell_size,
                label=f"same {same2.parent.name}",
            )
        )

        row_cells.append(
            make_gallery_partial_cam_cell(
                explainer=explainer,
                pre=pre,
                query_img_path=query_img,
                partial_img_path=diff1,
                mean=mean,
                std=std,
                cell_size=cell_size,
                label=f"diff {diff1.parent.name}",
            )
        )

        row_cells.append(
            make_gallery_partial_cam_cell(
                explainer=explainer,
                pre=pre,
                query_img_path=query_img,
                partial_img_path=diff2,
                mean=mean,
                std=std,
                cell_size=cell_size,
                label=f"diff {diff2.parent.name}",
            )
        )

        grid_rows.append(row_cells)

    explainer.remove()

    grid_rgb = stack_grid(grid_rows)
    save_rgb(out_path, grid_rgb)

    print(f"[INFO] saved grid to: {out_path}")


if __name__ == "__main__":
    run_dual_encoder_grid_gradcam(
        fullface_ckpt_path="work_dirs/exp_glint360k_roi_100_r50_arcface/best_model.pt",
        partialface_ckpt_path="work_dirs/clip_ratio_20/best_model.pt",
        fullface_base_dir="/datasets/glint360k/ROIs/ratio_100/test/",
        partialface_base_dir="/datasets/glint360k/ROIs/ratio_20/test/",
        out_path="outputs/dual_encoder_pair_gradcam/grid_5x5.jpg",
        network_full="r50",
        network_partial="vit_s_dp005_mask_0",
        device="cuda",
        image_size=(112, 112),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        seed=42,
        cell_size=(224, 224),
    )