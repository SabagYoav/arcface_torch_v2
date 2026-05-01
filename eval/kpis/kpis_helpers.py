"""
# data is arranged as such:
- main_dir_1 (full faces)
    - class_1
        - img_1
        - img_2
        - ...
    - class_2
        - img_1
        - img_2
        - ...
- main_dir_2 (part faces)
    - class_1
        - img_1
        - img_2
        - ...
    - class_2
        - img_1
        - img_2
        - ...
"""

import os
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

import os
from typing import List

def get_file_paths(dir_path: str, exts: List[str] = None, ext_case_sensitive: bool = False, recursive: bool = False, max_depth: int = -1) -> List[str]:
    """
    Returns all file paths from given directory
    Args:
        dir_path: directory path in question
        exts: extensions (expects a list with elements beginning with .)
        ext_case_sensitive: boolean - are the given exts case sensitive or not
        recursive: boolean, to query sub-folders as well
        max_depth: int. recursion level - how many levels of sub-folders are allowed (-1 [default] for all)
    Returns:
        list of file paths
    """
    if dir_path is None:
        raise ValueError("Invalid directory path (None)")
    if not isinstance(dir_path, str):
        raise ValueError(f"Invalid directory path type (expected str, got {type(dir_path)})")
    if not os.path.isdir(dir_path):
        raise ValueError(f"{dir_path} is not a directory or doesn't exist")
    while dir_path.endswith('/'):
        dir_path = dir_path[:-1]

    if not recursive:
        max_depth = 1

    result_paths = list()
    # r=root, d=directories, f = files
    for r, d, f in os.walk(dir_path):
        depth = r[len(dir_path):].count(os.sep)

        if max_depth >= 0:
            if depth >= max_depth:
                continue

        for file in f:
            result_paths.append(os.path.join(r, file))

    if exts:
        if ext_case_sensitive:
            result_paths = filter(lambda x: os.path.splitext(x)[1] in exts, result_paths)
        else:
            exts = [e.lower() for e in exts]
            result_paths = filter(lambda x: (os.path.splitext(x)[1]).lower() in exts, result_paths)
        result_paths = list(result_paths)

    return result_paths

from random import shuffle


class RandomPopPipe(list):
    """
    RandomPopPipe is a list that can pop items in random order, without repeat.
    When all items are used, it will "regenerate" the pipe.
    * non pop functions will work as a normal list
    * list data doesn't actually change when pop is called
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._idx_list = list()
        self.__init_idx_list()

    def __init_idx_list(self):
        if len(self._idx_list) == 0:
            self._idx_list = list(range(len(self)))
            shuffle(self._idx_list)

    def pop(self, idx: int = None):
        '''
        Pop a random item from the list.
        'regenerate' the list when it is empty.
        Args:
            idx: int, the index of the item to pop. If None, pop the last item.

        Returns: the popped item.
        '''
        if idx is None:
            idx = len(self._idx_list) - 1

        item = self[self._idx_list.pop(idx)]
        self.__init_idx_list()

        return item

IMAGE_EXTS = [".png", ".jpg", ".jpeg"]


class YoavDataset(Dataset):
    def __init__(self, main_dirs: list[str], items_per_class: int = 1):
        """
            Dataset that on each __getitem__ returns items_per_class images from each main dir, all from the same class.
            E.g., if main_dirs = [dir1, dir2], items_per_class=2, then each __getitem__ returns 4 images:
                - 2 images from dir1 of class X
                - 2 images from dir2 of class X
            The class X is chosen randomly on each __getitem__ call, without repeat, until all classes are used.
            Then the process repeats.
            :param main_dirs: list of main directories, each containing class subdirectories.
            :param items_per_class: number of items to sample per class from each main dir.
            :param batch_size: batch size (used to make sure you don't load same class multiple times in same batch).
        """

        # assert classes
        classes = [os.listdir(d) for d in main_dirs]
        [c_list.sort() for c_list in classes]
        assert all(c_list == classes[0] for c_list in classes), "Classes in main dirs are not the same"

        # class variables
        # self.transform = transforms.ToTensor()
        self.transform = transforms.Compose([
        transforms.Resize((112, 112)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

        # load data
        class_names = os.listdir(main_dirs[0])
        self.data_by_class_name = dict()
        for class_name in class_names:
            paths = get_file_paths(dir_path=os.path.join(main_dirs[0], class_name), exts=IMAGE_EXTS, recursive=True)
            rel_paths = [os.path.relpath(p, main_dirs[0]) for p in paths]
            if len(rel_paths) >= items_per_class:
                self.data_by_class_name[class_name] = rel_paths
        self.class_names = RandomPopPipe(list(self.data_by_class_name.keys()))
        self.class_names_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # user params
        self.main_dirs = main_dirs
        self.items_per_class = min(items_per_class, min(len(v) for v in self.data_by_class_name.values()))

    def __len__(self):
        return len(self.class_names)

    def __getitem__(self, idx):
        # choose class
        class_name = self.class_names.pop(idx=0)

        # prepare all paths
        all_paths = list()
        rel_paths = random.sample(population=self.data_by_class_name[class_name], k=self.items_per_class)
        [all_paths.extend([os.path.join(main_dir, p) for p in rel_paths]) for main_dir in self.main_dirs]

        # load all images
        data = torch.stack([self.transform(Image.open(p)) for p in all_paths], dim=0) #TODO: check 

        # prep labels
        labels = torch.tensor([self.class_names_to_idx[class_name]] * len(all_paths), dtype=torch.long)

        return data, labels
    
    def collate_fn(self, batch):
        data = torch.vstack([item[0] for item in batch])
        labels = torch.hstack([item[1] for item in batch])
        return data, labels

def unnormalize(img_tensor, mean, std):
    # img_tensor: [B, C, H, W] or [C, H, W]
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return img_tensor * std + mean

def visualize_batch(batch_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], nrow=8):
    """
    Visualize a batch of images.
    Args:
        batch_tensor: [B, C, H, W] tensor
        mean: mean used for normalization
        std: std used for normalization
        nrow: number of images in a row
    Returns:
        grid_img: PIL Image of the grid
    """
    from torchvision.utils import make_grid
    import torchvision.transforms as T

    # batch_tensor = batch_tensor.squeeze()
    # if batch_tensor.dim() == 5 and batch_tensor.shape[2] == 3:
    #     batch_tensor = batch_tensor.view(-1, 3, 112, 112)  # [16*3, 3, 112, 112]

    unnorm_batch = unnormalize(batch_tensor, mean, std)
    grid_tensor = make_grid(unnorm_batch, nrow=nrow, padding=2)
    grid_img = T.ToPILImage()(grid_tensor.clamp(0, 1))
    grid_img.save("grid.png")

    # return grid_img

def main():
    # main_dirs = [
    #     "/src/path/to/main_dir_1",
    #     "/src/path/to/main_dir_2",
    # ]
    main_dirs = [
        "/DATA/glint360k/imageFolder_split_fullface/test" 
    ]
    batch_size = 8
    dataset = YoavDataset(main_dirs=main_dirs, items_per_class=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    for batch_idx, data in enumerate(dataloader):
        print(f"Batch {batch_idx}: data shape = {data[0].shape}")
        visualize_batch(data[0], nrow=8)


if __name__ == '__main__':
    main()