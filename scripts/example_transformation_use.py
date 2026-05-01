from matplotlib import image
from torch.utils.data import DataLoader
import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
from torchvision.datasets import ImageFolder

transform = transforms.Compose([
        # ----------------------------------------
        # GROUP A: random color/blur/quality augs
        # ----------------------------------------
        transforms.RandomApply([
            transforms.RandomChoice([
                # transforms.ColorJitter(saturation=0.5, hue=.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            ], p=[0.6, 0.4]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomPosterize(bits=4, p=0.2),
        ], p=0.75),
    ])

class ScratchDataset():
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.data = os.listdir(root) # Load your data here
        self.length = len(self.data)    

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.data[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)



from torchvision.utils import make_grid   
from torchvision.transforms import ToPILImage
if __name__ == "__main__":

    folder_root = "/DATA/faces/arc2face"
    
    dataset = ImageFolder(root=folder_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

    to_pil = ToPILImage()

    for batch_num, batch in enumerate(dataloader):
        if batch_num >= 10:
            break
        # If batch is a tuple (inputs, labels), use batch[0]
        imgs = batch if isinstance(batch, torch.Tensor) else batch[0]
        # If images are not tensors, convert them
        if not isinstance(imgs, torch.Tensor):
            imgs = torch.stack([torch.from_numpy(img) for img in imgs])
        grid = make_grid(imgs, nrow=5, padding=2)
        img_grid = to_pil(grid)
        img_grid.save(f"grid_batch_{batch_num}.jpg")