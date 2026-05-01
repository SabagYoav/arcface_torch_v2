import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import OrderedDict, Tuple

os.sys.path.append('.')
from dataset import get_dataloader
from backbones import get_model
from utils.utils_config import get_config
# from tensor_boad_visualization import log_embeddings_to_tensorboard
from kpis_helpers import YoavDataset


class FaceVerificator:
    """
    Fast, vectorized CMC evaluation (Rank-1/5/10) compatible with your dataloader:
        for _, (imgs, labels) in enumerate(val_loader): ...
    Uses cosine similarity on L2-normalized embeddings.
    """

    def __init__(self, subsample, seed: int = 42):
        self.seed = seed
        self.plt_save_path = 'eval/plots/ver_scores_distributions_plot.png'
        self.subsample = subsample

    @torch.no_grad()
    def _collect_embeddings( self, backbone: torch.nn.Module, val_loader, device: str = "cuda" ):
        backbone.eval()
        dev = torch.device(device)
        embs, labels = [], []

        for _, (imgs, lbs) in enumerate(val_loader):
            if _ > 300:
                break
            imgs = imgs.to(dev, non_blocking=True)
            feats = backbone(imgs)
            # Cast to float32 for stable sim math (even if model runs fp16)
            feats = feats.float()
            feats = F.normalize(feats, p=2, dim=1)
            embs.append(feats.detach().cpu())
            labels.append(lbs.detach().cpu())

        all_embs = torch.cat(embs, dim=0)     # [N, D]
        all_labels = torch.cat(labels, dim=0) # [N]
        if self.subsample == True:
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

    def count_metrics(self, pairs: list, thresh: float) -> Tuple[int, int, int, int]:
        TP = TN = FP = FN = 0
        for (emb1, emb2), label, score in pairs:
            if label == "same":
                if score >= thresh:
                    TP += 1
                else:
                    FN += 1
            else:  # label == "diff"
                if score < thresh:
                    TN += 1
                else:
                    FP += 1
        return TP, TN, FP, FN
    
    def plot_histogram(self, same_scores, diff_scores):
        # Plot distributions
        plt.hist(same_scores, bins=50, alpha=0.7, label='Same')
        plt.hist(diff_scores, bins=50, alpha=0.7, label='Diff')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.title('Distribution of Same/Diff Pair Similarities')
        plt.legend()
        try:
            plt.savefig(self.plt_save_path)
            print(f"Saved verification scores distribution plot to {self.plt_save_path}")
        except Exception as e:
            plt.savefig("./ver_scores_distributions_plot.png")
            print(f"Saved verification scores distribution plot to ./ver_scores_distributions_plot.png")
        plt.close()

    def compute_verification_acc(self, backbone, dataloader, threshs, device='cuda'):
        embs, labels = self._collect_embeddings(backbone, dataloader, device)
        set_labels = set(labels.tolist())
        
        pairs = []
        same_scores = []
        diff_scores = []
        for lbl in tqdm(set_labels):
            indices = torch.where(labels == lbl)[0].tolist()
            if len(indices) < 2:
                continue
            same_embs = (embs[indices[0]], embs[indices[1]])
            same_score = torch.cosine_similarity(same_embs[0].unsqueeze(0), same_embs[1].unsqueeze(0)).item()
            same_scores.append(same_score)
            same_dist = (1 - same_score) / 2
            pairs.append((same_embs, "same", same_score))
            diff_lbl = np.random.choice(list(set_labels - {lbl}))
            diff_indices = torch.where(labels == diff_lbl)[0].tolist()
            diff_embs = (embs[indices[0]], embs[diff_indices[0]])
            diff_score = torch.cosine_similarity(diff_embs[0].unsqueeze(0), diff_embs[1].unsqueeze(0)).item()
            diff_scores.append(diff_score) 
            diff_dist = (1 - diff_score) / 2
            pairs.append((diff_embs, "diff", diff_score))
        
        #log 
        print(f"Evaluating on {len(set_labels)} identities.")
        print(f"calculated total pairs:", len(pairs))

        self.plot_histogram(same_scores, diff_scores)
        best_acc = 0.0
        best_TP = best_TN = best_FP = best_FN = 0
        for thresh in threshs:
            TP, TN, FP, FN = self.count_metrics(pairs, thresh)
            acc = (TP + TN) / (TP + TN + FP + FN)
            if acc > best_acc:
                best_acc = acc
                best_TP, best_TN, best_FP, best_FN = TP, TN, FP, FN
        
        print(f"Threshold: {thresh:.2f} | Accuracy: {best_acc:.4f} | TP: {best_TP}, TN: {best_TN}, FP: {best_FP}, FN: {best_FN}")
        
        return best_TP, best_TN, best_FP, best_FN

def init_params(cfg_path: str):
    ## Dataloader ##

    cfg = get_config(cfg_path)
    cfg.loss_fn = 'test ranks'
    test_data_root = cfg.rec.replace('/train', '/test')
    cfg.test_data_root = test_data_root
    best_model_path = os.path.join(cfg.output, "best_model.pt")

    #'/DATA/lfw/lfw_funneled'
    # dataloader = get_dataloader(cfg, root_dir=data_root, loss_fn=cfg.loss_fn, batch_size=144, num_workers=1, local_rank=0)
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
    ## directories set up
    torch.manual_seed(42); np.random.seed(42)
    # dataset = "glint360k"
    # loss = "arcface"
    # roi = "eyes_and_nose"
    threshs = [ 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3, 0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4, ]

     ## Init ##
    # data_root = f"/datasets/{dataset}/imageFolder_split_{roi}/test"
    # model_path = f"work_dirs/{dataset}/augmentation/{loss}/{roi}/model_best.pt"
    # model_path = "/media/yoav/Yoav/arcface_torch_v0/work_dirs/glint360k/no_augmentation/arcface/fullface/model_best.pt"
    # model_path = "work_dirs/config_glint360k_subset_eyes_and_nose/best_model.pt"
    cfg_path = "configs/exp_glint360k_roi_15_r50_arcface.py"
    cfg, dataloader, backbone = init_params(cfg_path = cfg_path)

    ## Eval ##
    evaluator = FaceVerificator(subsample=False) 
    evaluator.plt_save_path = f"{cfg.output}/ver_scores_distributions_plot.png"
    TP, TN, FP, FN= evaluator.compute_verification_acc(backbone, dataloader=dataloader, threshs = threshs, device='cuda')
    print(f"dataset: {cfg.data_root} | Accuracy: { (TP+TN)/(TP + FP + TN + FN)} | recall: {TP/(TP+FN)} | precision: {TP/(TP+FP)}")
    print("Done evaluation.")

    ## TensorBoard Embedding Projector
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    # log_embeddings_to_tensorboard(
    #     backbone=backbone,
    #     val_loader=dataloader,
    #     tb_logdir="work_dirs/emb_proj",
    #     device="cuda",
    #     max_points=10000,            # keep it reasonable; you can raise this later
    #     thumb_hw=(64, 64),          # thumbnail size
    #     metadata_from_labels=True,  # show label strings in the projector panel
    #     class_names=None,           # or provide {id: name}
    #     mean=mean, std=std,
    # )


