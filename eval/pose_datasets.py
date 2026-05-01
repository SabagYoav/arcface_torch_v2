from pathlib import Path
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class PoseFilteredDataset(Dataset):
    def __init__(self, df, transform=None, roi_prefix=""):
        """
        df: DataFrame with columns ['path', 'yaw', 'pitch', 'roll']
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # label = parent folder name (ImageFolder convention)
        self.labels = [
            int(Path(p).parent.name) for p in self.df["path"]
        ]

        self.roi_prefix = roi_prefix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["path"]
        #update path
        const_prefix = "/datasets/glint360k/"#'/Downloads/Yoav/datasets/glint360k/'
        # start = img_path.find('/test')
        # img_path = const_prefix + self.roi_prefix + img_path[start:] 
        img_path = os.path.join(const_prefix, self.roi_prefix, img_path)

        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
    
class CrossPosePairWithNegativeDataset(Dataset):
    def __init__(
        self,
        df_pose_a,
        df_pose_b,
        transform=None,
        roi_prefix="imageFolder_split_narrow_eyes",
        const_prefix="/datasets/glint360k",
        seed=0,
        neg_from_same_pose=True,  # sample negative from pose B
    ):
        self.transform = transform
        self.roi_prefix = roi_prefix
        self.const_prefix = const_prefix
        self.rng = random.Random(seed)
        self.neg_from_same_pose = neg_from_same_pose

        # --- extract labels ---
        df_pose_a = df_pose_a.copy()
        df_pose_b = df_pose_b.copy()

        df_pose_a["label"] = df_pose_a["path"].apply(
            lambda p: int(Path(p).parent.name)
        )
        df_pose_b["label"] = df_pose_b["path"].apply(
            lambda p: int(Path(p).parent.name)
        )

        # --- group by identity ---
        self.group_a = {
            k: v["path"].tolist()
            for k, v in df_pose_a.groupby("label")
        }
        self.group_b = {
            k: v["path"].tolist()
            for k, v in df_pose_b.groupby("label")
        }

        # --- shared identities only ---
        self.labels = sorted(set(self.group_a) & set(self.group_b))

    def __len__(self):
        return len(self.labels)

    def _load(self, rel_path):
        img_path = os.path.join(
            self.const_prefix,
            self.roi_prefix,
            rel_path,
        )
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        # ---------- positive identity ----------
        label = self.labels[idx]

        path_a_pos = self.rng.choice(self.group_a[label])
        path_b_pos = self.rng.choice(self.group_b[label])

        img_a_pos = self._load(path_a_pos)
        img_b_pos = self._load(path_b_pos)

        # ---------- negative identity ----------
        neg_label = self.rng.choice(
            [l for l in self.labels if l != label]
        )

        if self.neg_from_same_pose:
            path_b_neg = self.rng.choice(self.group_b[neg_label])
        else:
            path_b_neg = self.rng.choice(self.group_a[neg_label])

        img_b_neg = self._load(path_b_neg)

        return (
            img_a_pos,
            img_b_pos,
            img_b_neg,
            torch.tensor(label, dtype=torch.long),
        )
def get_loader(df, batch_size=8, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = PoseFilteredDataset(df, transform, roi_prefix="imageFolder_split_narrow_eyes")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,      # IMPORTANT for verification
        num_workers=num_workers,
        pin_memory=True
    )

def get_cross_pose_pair_loader(
    df_pose_a,
    df_pose_b,
    batch_size=8,
    num_workers=2,
    shuffle=False,  # usually False for verification
):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    dataset = CrossPosePairWithNegativeDataset(
        df_pose_a,
        df_pose_b,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )