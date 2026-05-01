"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import os
import pickle
from PIL import Image

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# import mxnet as mx
import numpy as np
import sklearn
import torch
# from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from torchvision import transforms
from collections import defaultdict
import random
import torch
import torch.nn.functional as F


class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            print("x:", far_train)
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print(true_accept, false_accept)
    # print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list

def convers_tensor_to_triplet(tensors, labels):
    labels_set = set([lbl.item() for lbl in labels])
    if len(labels_set) < 2:
        return []
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label.item()].append(idx)

    triplets = []
    for idx, label in enumerate(labels):
        anchor = tensors[idx]
        positive_idx = idx
        positive_idx = random.choice(label_to_indices[label.item()])
        positive = tensors[positive_idx]

        negative_label = random.choice(list(labels_set - set([label.item()])))
        negative_idx = random.choice(label_to_indices[negative_label])
        negative = tensors[negative_idx]

        triplets.append((anchor, positive, negative))
    return triplets


def test_triplet_loss(backbone: torch.nn.Module, dataloader: DataLoader, max_triplets=500):
    backbone.eval()
    total_loss = 0.0
    total_triplets = 0

    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(dataloader):
            if i * dataloader.batch_size >= max_triplets:
                break
            triplets = convers_tensor_to_triplet(imgs, lbls)

            for anchor, positive, negative in triplets:
                anchor   = anchor.unsqueeze(0).cuda()
                positive = positive.unsqueeze(0).cuda()
                negative = negative.unsqueeze(0).cuda()

                emb_a = F.normalize(backbone(anchor), dim=1)
                emb_p = F.normalize(backbone(positive), dim=1)
                emb_n = F.normalize(backbone(negative), dim=1)

                loss = F.triplet_margin_loss( emb_a, emb_p, emb_n,margin=0.2, p=2)

                total_loss += loss.item()
                total_triplets += 1

    return total_loss / max(total_triplets, 1)

# @torch.no_grad()
# def load_image_folder(path, image_size=(112,112)):
#     transform = transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.ToTensor(),  # Converts to [0,1] CHW float32
#     ])
#     imgs = []
#     labels = []
#     label_map = {}   # folder_name -> class_index
    
#     # ----------- iterate subfolders -----------
#     for folder_name in sorted(os.listdir(path)):
#         folder_path = os.path.join(path, folder_name)
#         class_id = int(folder_name)

#         if not os.path.isdir(folder_path):
#             continue

#         # loop inside folder
#         for fname in sorted(os.listdir(folder_path)):
#             fpath = os.path.join(folder_path, fname)

#             if not fpath.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
#                 continue

#             # img = Image.open(fpath).convert("RGB")
#             # img_t = transform(img)
            
#             imgs.append(fpath)
#             labels.append(class_id)

    
#     # stack list into tensor
#     # imgs = torch.stack(imgs)  # [N, 3, H, W]

#     print(f"Loaded {len(imgs)} images from {path}")
#     print(f"Number of classes: {len(label_map)}")

#     return imgs, labels
import random
from torch.utils.data import Subset

def subsample_dataset_by_ids(dataset, num_ids=1000, seed=42):
    random.seed(seed)

    # Get all unique class IDs
    labels = dataset.targets  # list[int]
    unique_ids = list(set(labels))

    if num_ids is None or num_ids >= len(unique_ids):
        return dataset, set(unique_ids)
    selected_ids = set(random.sample(unique_ids, num_ids))

    # Keep only indices whose label is in selected_ids
    indices = [i for i, y in enumerate(labels) if y in selected_ids]

    return Subset(dataset, indices), selected_ids

@torch.no_grad()
def load_image_folder(
    path,
    image_size=(112, 112),
    batch_size=64,
    subset_num_ids=None,
    subset_seed=42,
):
    dataset = ImageFolder(path, transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]))

    if subset_num_ids is not None:
        dataset, _ = subsample_dataset_by_ids(dataset, num_ids=subset_num_ids, seed=subset_seed)

    # num_workers=0: eval DataLoaders are persistent for all epochs; workers fork with the
    # full ImageFolder (1.4M entries) in memory. With num_workers=4 that costs ~280MB × 4 × 2
    # dataloaders = ~2.2GB just in worker processes, filling swap.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    return dataloader

@torch.no_grad()
def test_bin(data_set, backbone, batch_size, nfolds=10):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list

@torch.no_grad()
def test_image_dataloader(data_set, backbone, batch_size, nfolds=10):
    transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),  # Converts to [0,1] CHW float32
    ])
    max_ids = 1000
    max_per_id = 10 #TODO : implement before next run, currently not used
    print('testing verification..')
    images = data_set[0]
    labels = data_set[1]

    same_class_pairs = []
    diff_class_pairs = []
    embeddings_list = []

    for i in range(min(len(images), max_ids)): #TODO: batching is needed
        img = transform(Image.open(images[i]).convert("RGB"))
            
        emb = backbone(img.unsqueeze(0))
        norm_embs = emb / torch.norm(emb, p=2, dim=1, keepdim=True)
        embeddings_list.append(norm_embs.to("cpu"))
    
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            if labels[i] == labels[j]:
                same_class_pairs.append((embeddings_list[i], embeddings_list[j], 'True', F.cosine_similarity(embeddings_list[i], embeddings_list[j])))
            else:
                diff_class_pairs.append((embeddings_list[i], embeddings_list[j], 'False',F.cosine_similarity(embeddings_list[i], embeddings_list[j])))

    diff_class_balanced = random.sample(diff_class_pairs, len(same_class_pairs))
    combined = same_class_pairs + diff_class_balanced
    
    threshold = 0.5
    true_positive, false_negative, false_positive, true_negative = 0, 0, 0, 0
    for item in combined:
        if item[2] == 'True' and item[3] > threshold:
            true_positive += 1
        elif item[2] == 'True' and item[3] <= threshold:
            false_negative += 1
        elif item[2] == 'False' and item[3] > threshold:
            false_positive += 1
        else:
            true_negative += 1
    accuracy = (true_positive + true_negative) / len(combined)
    return accuracy, embeddings_list


def test_fold(embeddings, labels):

    E = torch.cat(embeddings, dim=0)  # [N, D]
    L = torch.cat(labels, dim=0)      # [N]

    # cosine similarity matrix — run on GPU to avoid saturating all CPU BLAS threads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E = E.to(device)
    L = L.to(device)
    S = E @ E.T

    same = L.unsqueeze(0) == L.unsqueeze(1)
    diag = ~torch.eye(len(L), dtype=torch.bool, device=device)

    pos = S[same & diag]
    neg = S[~same & diag]

    # balance
    w_pos = 1.0
    w_neg = len(pos) / len(neg)

    # thresholds = torch.linspace(0.2, 0.9, 50)
    #thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = [0.2]
    best_acc, best_thr, best_tar, best_tar_far = 0.0, 0.0, 0.0, 0.0

    for t in thresholds:
        tp = (pos > t).sum().float()
        fn = (pos <= t).sum().float()
        tn = (neg <= t).sum().float()
        fp = (neg > t).sum().float()
        # acc = (tp + tn).float() / (len(pos) + len(neg))
        acc = (w_pos * tp + w_neg * tn) / (w_pos * (tp + fn) + w_neg * (tn + fp))
        tar = tp/(tp + fn)

        if acc > best_acc:
            best_acc = acc.item()
            best_thr = t
        if tar > best_tar:  
            best_tar = (tp/(tp + fn)).item()
            best_tar_far = (fp/(fp + tn)).item()
    return best_acc, best_thr, best_tar, best_tar_far
    

@torch.no_grad()
def test_image_dataloader_with_fold(dataloader, backbone, max_fold_images=500, k_fold=10):
    backbone.eval()
    embeddings = []
    labels = []
    accs, threshs = [], []
    fold = 0

    with torch.no_grad():
        for imgs, lbls in (dataloader):
            if fold>k_fold:
                break
            imgs = imgs.cuda()
            emb = F.normalize(backbone(imgs), dim=1)
            embeddings.append(emb.cpu())
            labels.append(lbls.cpu())

            if sum(e.shape[0] for e in embeddings) >= max_fold_images:
                acc, thresh, tar, far = test_fold(embeddings, labels)
                accs.append(acc)
                threshs.append(thresh)
                embeddings, labels = [], []
                fold += 1
        
        if len(accs) == 0 and len(embeddings) > 0:
            acc, thresh, tar, far = test_fold(embeddings, labels)
            accs.append(acc)
            threshs.append(thresh)

    mean_acc = np.mean(accs)
    best_acc = np.max(accs)
    best_thr = threshs[ accs.index(best_acc) ]  

    #print(f"[DEBUG] Best thr={best_thr:.3f}, acc={best_acc:.4f}")
    
    return mean_acc, best_thr, tar, far

def dumpR(data_set,
          backbone,
          batch_size,
          name='',
          data_extra=None,
          label_shape=None):
    print('dump verification embedding..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba

            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra),
                                     label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    actual_issame = np.asarray(issame_list)
    outname = os.path.join('temp.bin')
    with open(outname, 'wb') as f:
        pickle.dump((embeddings, issame_list),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    ## test verification.load_img_folder functino
    data_set = load_img_folder('/DATA/faces/vggfaces2/overfit/test')
