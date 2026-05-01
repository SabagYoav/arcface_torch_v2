import torch
import torch.nn.functional as F
from typing import Tuple
import os
import argparse
import numpy as np
from tqdm import tqdm
os.sys.path.append('.')
from dataset import get_dataloader
from backbones import get_model
from utils.utils_config import get_config
from collections import Counter

from dataset import YoavDataset

# from tensor_boad_visualization import log_embeddings_to_tensorboard
from collections import OrderedDict

# from utils.custom_dataloader import YoavDataset

class FaceIditificator:
    """
    Fast, vectorized CMC evaluation (Rank-1/5/10) compatible with your dataloader:
        for _, (imgs, labels) in enumerate(val_loader): ...
    Uses cosine similarity on L2-normalized embeddings.
    """

    def __init__(self, max_k: int = 10, normalize: bool = True,  max_classes: int = None, max_instances: int = None, seed: int = 42):
        self.max_k = max_k
        self.normalize = normalize
        self.max_classes = max_classes
        self.max_instances = max_instances
        self.seed = seed

    @torch.no_grad()
    def _collect_embeddings( self, backbone: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: str = "cuda" ) -> Tuple[torch.Tensor, torch.Tensor]:
        backbone.eval()
        dev = torch.device(device)
        embs, labels = [], []

        for _, (imgs, lbs) in enumerate(val_loader):

            imgs = imgs.to(dev, non_blocking=True)
            ## if YoavDataset returns
            imgs = imgs.squeeze(2)  # Remove singleton dimension: [16, 2, 3, 112, 112]
            imgs = imgs.view(-1, 3, 112, 112)  # Flatten batch and image count: [32, 3, 112, 112]
            #######################
            feats = backbone(imgs)
            # Ensure 2D [B, D]
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)
            # Cast to float32 for stable sim math (even if model runs fp16)
            feats = feats.float()
            if self.normalize:
                feats = F.normalize(feats, p=2, dim=1)
            embs.append(feats.detach().cpu())
            if isinstance(lbs, list) or isinstance(lbs, tuple):
                lbs = [int(l) for l in lbs]
                lbs = torch.tensor(lbs)
            labels.append(lbs.detach().cpu())
        all_embs = torch.cat(embs, dim=0)     # [N, D]
        all_labels = torch.cat(labels, dim=0) # [N]
        all_embs, all_labels = self._subsample(all_embs, all_labels)
        return all_embs, all_labels

    def _subsample(self, embs, labels):
        """
        Limit number of classes and instances per class.
        """
        rng = torch.Generator().manual_seed(self.seed)
        chosen_embs, chosen_labels = [], []

        # Get unique labels
        unique_labels = torch.unique(labels)

        # If max_classes is set, randomly pick a subset
        if self.max_classes is not None and len(unique_labels) > self.max_classes:
            chosen_classes = unique_labels[torch.randperm(len(unique_labels), generator=rng)[:self.max_classes]]
        else:
            chosen_classes = unique_labels

        for cls in chosen_classes:
            idxs = (labels == cls).nonzero(as_tuple=True)[0]
            if self.max_instances is not None and len(idxs) > self.max_instances:
                idxs = idxs[torch.randperm(len(idxs), generator=rng)[:self.max_instances]]
            chosen_embs.append(embs[idxs])
            chosen_labels.append(labels[idxs])

        return torch.cat(chosen_embs, 0), torch.cat(chosen_labels, 0)


    @torch.no_grad()
    def compute_retrieval_ranks(
        self,
        backbone,
        dataloader,
        device: str = "cuda",
        qbs: int = 2048,     # query block size
        gbs: int = 32768,    # gallery block size
    ):
        """

        """
        backbone.eval()
        query_embs, gallery_embs = [], []
        embs, labels = self._collect_embeddings(backbone, dataloader, device=device)
        set_labels = set(labels.tolist())
        for lbl in tqdm(set_labels):
            indices = torch.where(labels == lbl)[0].tolist()
            query_emb = embs[indices[0]]
            query_embs.append(query_emb.unsqueeze(0))
            gallery_embs.append(embs[indices[1:]] if len(indices) > 1 else torch.empty(0))
        query_embs=torch.stack(query_embs).squeeze(1)  # [num_ids, D]
        gallery_embs = torch.cat(gallery_embs, dim=0) 
        print(f"Evaluating on {len(set_labels)} identities, gallery size {len(gallery_embs)} samples.")

        ## Compute ranks ##
        r1, r5, r10 = 0.0, 0.0, 0.0
        for query_emb, lbl in tqdm(zip(query_embs, labels)):
            scores = torch.cosine_similarity(query_emb, gallery_embs).cpu()
            sorted_indices = torch.argsort(scores, descending=True)  # indices of gallery_embs sorted by similarity (max to min)
            sorted_labels = labels[sorted_indices]
            if lbl in sorted_labels[:1]:
                r1 += 1
            if lbl in sorted_labels[:5]:
                r5 += 1
            if lbl in sorted_labels[:10]:
                r10 += 1
        return r1, r5, r10

    @torch.no_grad()
    def compute_retrieval_ranks_with_knn(
        self,
        backbone,
        dataloader,
        device="cuda",
        k_for_cmc=(1,5,10),
        k_for_knn=5,   # majority-vote KNN classification
    ):
        backbone.eval()
        embs, labels = self._collect_embeddings(backbone, dataloader, device=device)
        # embs should already be L2-normalized by _collect_embeddings()

        # Build query/gallery split: 1st per class = query, rest = gallery
        unique_labels = torch.unique(labels)
        query_embs, query_labels = [], []
        gallery_embs, gallery_labels = [], []

        for lbl in unique_labels:
            idx = torch.where(labels == lbl)[0]
            if len(idx) < 2:
                # skip singletons (no positive in gallery)
                continue
            q, g = idx[0], idx[1:]
            query_embs.append(embs[q].unsqueeze(0))
            query_labels.append(labels[q].unsqueeze(0))
            gallery_embs.append(embs[g])
            gallery_labels.append(labels[g])

        if len(query_embs) == 0:
            return {"rank@1":0.0, "rank@5":0.0, "rank@10":0.0, "knn@{}".format(k_for_knn):0.0}

        query_embs = torch.cat(query_embs, dim=0)        # [Q, D]
        query_labels = torch.cat(query_labels, dim=0)    # [Q]
        gallery_embs = torch.cat(gallery_embs, dim=0)    # [G, D]
        gallery_labels = torch.cat(gallery_labels, dim=0)# [G]

        # Cosine scores for ALL queries at once (since embs are L2-normalized, dot = cosine)
        # scores[q, g] = cosine sim
        scores = query_embs @ gallery_embs.t()           # [Q, G]

        # ---- CMC / Rank-k (hits@k) ----
        cmc = {}
        for k in k_for_cmc:
            topk_idx = scores.topk(k, dim=1).indices     # [Q, k]
            topk_labs = gallery_labels[topk_idx]         # [Q, k]
            hit = (topk_labs == query_labels.view(-1,1)).any(dim=1).float().mean().item() * 100.0
            cmc[f"rank@{k}"] = hit

        # ---- KNN classification (majority vote) ----
        k = min(k_for_knn, gallery_embs.size(0))
        knn_idx = scores.topk(k, dim=1).indices          # [Q, k]
        knn_labs = gallery_labels[knn_idx]               # [Q, k]

        preds = []
        for row in knn_labs.tolist():
            # majority vote; break ties by highest-sim label (optional: weighted vote)
            c = Counter(row)
            pred = c.most_common(1)[0][0]
            preds.append(pred)
        preds = torch.tensor(preds, dtype=query_labels.dtype)

        knn_acc = (preds == query_labels).float().mean().item() * 100.0

        return {**cmc}
        
def init_params(cfg_path: str):
    ## Dataloader ##

    cfg = get_config(cfg_path)
    cfg.loss_fn = 'test ranks'
    test_data_root = cfg.rec.replace('/train', '/test')
    cfg.test_data_root = test_data_root
    best_model_path = os.path.join(cfg.output, "best_model.pt")

    dataset = YoavDataset(main_dirs=[test_data_root], items_per_class=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=72, shuffle=False, collate_fn=dataset.collate_fn)
    ## Model ##
    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size ).cuda()
    ckpt = torch.load(best_model_path)
    if "state_dict_backbone" in ckpt:
        backbone.load_state_dict(ckpt["state_dict_backbone"])
        backbone = backbone.cuda()
    elif isinstance(ckpt, OrderedDict):
        backbone.load_state_dict(ckpt)
        backbone = backbone.cuda()

    return cfg, dataloader, backbone


if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)
    dataset = "glint360k" # glint360k
    loss = "arcface"
    roi = "narrow_eyes"
    print(f"Evaluating dataset: {dataset}, loss: {loss}, roi: {roi} , augmentation: Yes")
     ## Init ##
    data_root = f"/datasets/{dataset}/imageFolder_split_{roi}/test"
    model_path = f"work_dirs/{dataset}/augmentation/{loss}/{roi}/model_best.pt"
    model_path = "work_dirs/glint360k_subset_narrow_eyes_vit_s/best_model.pt"
    # model_path = "work_dirs/16backbone.pth"
    cfg_path = "configs/glint360k_subset_fullface_vit_s.py"
    args, dataloader, backbone = init_params(cfg_path = cfg_path, data_root = data_root, model_path = model_path)

    ## Eval ##
    evaluator = FaceIditificator(max_classes=1000, max_instances=None) 
    # r1, r5, r10 = evaluator.compute_retrieval_ranks(backbone, dataloader=dataloader)
    # print(f"Rank-1: {r1:.2f}%  Rank-5: {r5:.2f}%  Rank-10: {r10:.2f}%")
    threshs = [ 0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3 ]
    results = evaluator.compute_retrieval_ranks_with_knn(backbone, dataloader=dataloader)
    print(results)

    print("Done")
    
    # mean = [0.5, 0.5, 0.5] /home/yoav/Downloads/Yoav/datasets/glint360k/imageFolder_split_single_narrow_eye/test/
    # std = [0.5, 0.5, 0.5]

    # log_embeddings_to_tensorboard(
    #     backbone=backbone,
    #     val_loader=dataloader,
    #     tb_logdir="work_dirs/emb_proj",
    #     device="cuda",
    #     max_points=4000,            # keep it reasonable; you can raise this later
    #     thumb_hw=(64, 64),          # thumbnail size
    #     metadata_from_labels=True,  # show label strings in the projector panel
    #     class_names=None,           # or provide {id: name}
    #     mean=mean, std=std,
    # )


