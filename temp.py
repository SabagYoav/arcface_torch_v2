from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images_a = list(Path(root_dir,'imageFolder_split_narrow_eyes/val').glob('*/*'))
        self.images_b = list(Path(root_dir,'imageFolder_split_fullface/val').glob('*/*'))

        self.transform = transform

    def __len__(self):
        return len(self.images_a)

    def __getitem__(self, idx):
        path_a = self.images_a[idx]
        path_b = self.images_b[idx]

        img_a = Image.open(path_a).convert("RGB")
        img_b = Image.open(path_b).convert("RGB")

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        all_images = torch.stack([img_a, img_b], dim=0)
        all_labels = torch.tensor([int(path_a.parent.name), int(path_b.parent.name)])
        
        return all_images, all_labels

def paired_collate(batch):
    imgs, labels = zip(*batch)
    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    return imgs, labels

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataloader = torch.utils.data.DataLoader(
    PairedImageDataset(
        root_dir='/Downloads/Yoav/datasets/glint360k',
        transform=transform
    ),
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=paired_collate
)
for batch in dataloader:
    images, labels = batch
    print(images.shape, labels.shape)
    break