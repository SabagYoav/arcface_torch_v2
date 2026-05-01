import os
import torch
from torchvision.utils import make_grid, save_image

def save_batch_grid(
    img: torch.Tensor,
    save_path: str = "batch_grid.jpg",
    nrow: int = 16,
    normalize_from_minus1_1: bool = True,
):
    """
    img: [B, 3, H, W]
    """
    x = img.detach().cpu().clone()

    # undo Normalize(mean=0.5, std=0.5)  ->  [-1,1] to [0,1]
    if normalize_from_minus1_1:
        x = x * 0.5 + 0.5

    x = torch.clamp(x, 0.0, 1.0)

    grid = make_grid(x, nrow=nrow, padding=2)
    save_image(grid, save_path)
    print(f"saved grid to: {save_path}")