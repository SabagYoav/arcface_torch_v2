import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
    

def get_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 0,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Image Folder
    transform = get_transform(augmentations=True)

    train_set = ImageFolder(root_dir, transform)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader

def get_transform(augmentations=False):
    if augmentations:
            transform = transforms.Compose([
            # ----------------------------------------
            # GROUP A: random color/blur/quality augs
            # ----------------------------------------
            transforms.RandomApply([
            transforms.ColorJitter( brightness=0.2,  contrast=0.2,  saturation=0.2 ),
            transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
        ], p=0.2),
            # transforms.RandomApply([
            #     # transforms.RandomChoice([
            #         # transforms.ColorJitter(saturation=0.5, hue=.3),
            #         # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            #         # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            #     # ], p=[0.5, 0.3, 0.2]),
            #     transforms.RandomAdjustSharpness(sharpness_factor=1, p=0.15),
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     # transforms.RandomPosterize(bits=4, p=0.2),
            # ], p=0.50),

            # ----------------------------------------
            # ALWAYS-ON TRANSFORMS
            # ----------------------------------------
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    return transform

def get_paris_dataloader(
    dataset_type,
    root_dir,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 0,
    ) -> Iterable:

    rec = os.path.join(root_dir, 'train.rec')
    idx = os.path.join(root_dir, 'train.idx')
    train_set = None

    # Image Folder
    transform = get_transform(augmentations=True)

    if dataset_type == 'image_folder':
        train_set = ImageFolder(root_dir, transform)
    elif dataset_type == 'paired_image_folder':
        train_set = PairedImageDataset(root_dir, transform)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    if dataset_type == 'image_folder':
        train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=init_fn,
        )
    elif dataset_type == 'paired_image_folder':
        train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=init_fn,
            collate_fn=paired_collate
        )

    return train_loader


from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import random

from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

def get_clip_dataloader(
    root_pf,
    root_ff,
    local_rank,
    batch_size,
    dali = False,
    dali_aug = False,
    seed = 2048,
    num_workers = 0,
    ) -> Iterable:

    train_set = ClipDataset(root_pf=root_pf, root_ff=root_ff, transform=get_transform(augmentations=True))

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
        collate_fn=clip_paired_collate
    )
    return train_loader

class ClipDataset(Dataset):
    def __init__(self, root_pf, root_ff, transform=None):
        self.root_pf = Path(root_pf)
        self.root_ff = Path(root_ff)

        self.pairs = []
        for pf in self.root_pf.glob("*/*"):
            ff = self.root_ff / pf.relative_to(self.root_pf)
            if ff.exists():
                self.pairs.append((pf, ff))
        assert len(self.pairs) > 0
        self.transform = transform

        self.labels = [p[0].parent.name for p in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pf, ff = self.pairs[idx]
        assert pf.parent.name == ff.parent.name, f"Mismatched pair: {pf} and {ff}"
        lbl = int(pf.parent.name)

        img_pf = Image.open(pf).convert("RGB")
        img_ff = Image.open(ff).convert("RGB")

        if self.transform:
            img_pf = self.transform(img_pf)
            img_ff = self.transform(img_ff)

        return img_ff, img_pf, lbl
    
def clip_paired_collate(batch):
    partial, full, lbls = zip(*batch)
    return torch.stack(partial), torch.stack(full), torch.tensor(lbls)
    
    
class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # self.images_a = list(Path(root_dir,'imageFolder_split_narrow_eyes/train').glob('*/*'))
        # self.images_b = list(Path(root_dir,'imageFolder_split_fullface/train').glob('*/*'))

        ## filter b to match a
        # self.root_a = Path(root_dir, '/datasets/variants_dataset/train')
        # self.root_b = Path(root_dir, '/datasets/glint360k/imageFolder_split_fullface/train')
        self.root_a = Path('/datasets/variants_dataset/train')
        self.root_b = Path('/datasets/glint360k/imageFolder_split_fullface/train')

        self.images_a = list(self.root_a.glob('*/*'))
        self.images_b = list(self.root_b.glob('*/*'))

        rel_a = {
                    p.relative_to(self.root_a)
                    for p in self.images_a
                }
        self.images_b = [
                            p for p in self.images_b
                            if p.relative_to(self.root_b) in rel_a
                        ]
        
        assert len(self.images_a) == len(self.images_b), "Mismatched paired dataset lengths"

        self.transform = transform

    def __len__(self):
        return len(self.images_a)

    def __getitem__(self, idx):
        path1_a = self.images_a[idx]
        path1_b = self.images_b[idx]

        path2_a = random.choice(list(self.root_a.glob(path1_a.parent.name + '/*')))
        path2_b = self.root_b / path2_a.relative_to(self.root_a)

        img1_a = Image.open(path1_a).convert("RGB")
        img1_b = Image.open(path1_b).convert("RGB")
        img2_a = Image.open(path2_a).convert("RGB")
        img2_b = Image.open(path2_b).convert("RGB")

        # grid = make_grid([img1_a, img1_b, img2_a, img2_b], nrow=2, padding=2)
        # to_pil = ToPILImage()
        # grid_img = to_pil(grid)
        # grid_img.save("grid.jpg")


        if self.transform:
            img1_a = self.transform(img1_a)
            img1_b = self.transform(img1_b)
            img2_a = self.transform(img2_a)
            img2_b = self.transform(img2_b)
        
        all_images = torch.stack([img1_a, img1_b, img2_a, img2_b], dim=0)
        all_labels = torch.tensor([int(path1_a.parent.name), int(path1_b.parent.name), int(path2_a.parent.name), int(path2_b.parent.name)])
        
        return all_images, all_labels

def paired_collate(batch):
    imgs, labels = zip(*batch)
    imgs = torch.cat(imgs, dim=0)
    labels = torch.cat(labels, dim=0)
    # imgs = torch.stack(imgs, dim=0)    # shape: [batch_size, 4, 3, 112, 112]
    # labels = torch.stack(labels, dim=0) 
    return imgs, labels


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


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
    
from typing import List
def get_file_paths(dir_path: str, exts: list[str] = None, ext_case_sensitive: bool = False, recursive: bool = False, max_depth: int = -1) -> List[str]:
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
    
import os
import random
from collections import defaultdict
from functools import partial
from typing import Iterable, Iterator, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

# keep using your existing functions/classes if they already exist
# from dataset import DataLoaderX, get_transform, worker_init_fn
# from utils.utils_distributed_sampler import get_dist_info


class PKSampler(Sampler):
    """
    PK sampler for metric learning.

    Each batch contains:
        P identities
        K images per identity

    Total batch size = P * K

    Works with torchvision.datasets.ImageFolder because it uses dataset.targets.
    Supports distributed training by splitting the list of generated batches
    across ranks.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        num_instances: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 2048,
        drop_last: bool = True,
    ):
        if not hasattr(dataset, "targets"):
            raise ValueError("PKSampler requires dataset.targets (ImageFolder provides this).")

        if batch_size % num_instances != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by num_instances ({num_instances})."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.index_dic = defaultdict(list)
        for index, pid in enumerate(dataset.targets):
            self.index_dic[int(pid)].append(index)

        self.pids = list(self.index_dic.keys())

        if len(self.pids) < self.num_pids_per_batch:
            raise ValueError(
                f"Number of identities ({len(self.pids)}) is smaller than "
                f"num_pids_per_batch ({self.num_pids_per_batch})."
            )

        self.length = self._estimate_length()

    def _estimate_length(self) -> int:
        total = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            n = len(idxs)
            if n < self.num_instances:
                n = self.num_instances
            total += n - (n % self.num_instances)

        num_batches = total // self.batch_size
        if not self.drop_last and total % self.batch_size != 0:
            num_batches += 1

        local_num_batches = num_batches // self.world_size
        if not self.drop_last and num_batches % self.world_size != 0:
            local_num_batches += 1

        return local_num_batches * self.batch_size

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)

        # build chunks of K samples for each identity
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = self.index_dic[pid].copy()

            if len(idxs) < self.num_instances:
                idxs = np.random.RandomState(self.seed + self.epoch + pid).choice(
                    idxs, size=self.num_instances, replace=True
                ).tolist()
            else:
                rng.shuffle(idxs)

            pid_batches = []
            for i in range(0, len(idxs), self.num_instances):
                chunk = idxs[i:i + self.num_instances]
                if len(chunk) == self.num_instances:
                    pid_batches.append(chunk)

            batch_idxs_dict[pid] = pid_batches

        available_pids = [pid for pid in self.pids if len(batch_idxs_dict[pid]) > 0]
        final_batches: List[List[int]] = []

        while len(available_pids) >= self.num_pids_per_batch:
            selected_pids = rng.sample(available_pids, self.num_pids_per_batch)
            batch = []

            for pid in selected_pids:
                batch.extend(batch_idxs_dict[pid].pop(0))
                if len(batch_idxs_dict[pid]) == 0:
                    available_pids.remove(pid)

            if len(batch) == self.batch_size:
                final_batches.append(batch)

        # distributed split at batch level
        final_batches = final_batches[self.rank::self.world_size]

        # flatten batches into index stream
        final_indices = [idx for batch in final_batches for idx in batch]

        return iter(final_indices)


def get_batch_triplet_dataloader(
    root_dir,
    local_rank,
    batch_size,
    dali=False,
    dali_aug=False,
    seed=2048,
    num_workers=0,
    num_instances=4,   # K
) -> Iterable:
    """
    PK-sampler dataloader for ImageFolder.

    Args:
        root_dir: ImageFolder root
        local_rank: gpu rank for DataLoaderX
        batch_size: must be divisible by num_instances
        seed: random seed
        num_workers: dataloader workers
        num_instances: number of images per identity in a batch (K)

    Batch structure:
        P = batch_size // num_instances identities
        K = num_instances images per identity
    """

    # if you already have this in your project, keep your version
    rank, world_size = get_dist_info()

    transform = get_transform(augmentations=True)
    train_set = ImageFolder(root_dir, transform=transform)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_sampler = PKSampler(
        dataset=train_set,
        batch_size=batch_size,
        num_instances=num_instances,
        rank=rank,
        world_size=world_size,
        seed=seed,
        drop_last=True,
    )

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader