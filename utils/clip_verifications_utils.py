import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataset import get_clip_dataloader
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler


class ClipVerification(object):
    def __init__(
        self,
        val_targets, # [partial_root, full_root]
        train_targets,               
        summary_writer=None,
        image_size=(112, 112),
        batch_size=64,
        num_workers=1,
        wandb_logger=None,
        work_dir="./",
    ):
        # self.rank = 0 #distributed.get_rank()
        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger
        self.work_dir = Path(work_dir)
        self.train_partial_loader = None
        self.train_full_loader = None
        self.val_partial_loader = None
        self.val_full_loader = None
        self.current_epoch = -1

 
        self.val_loader = self.load_clip_dataloader(
            root_ff = val_targets[1], root_pf = val_targets[0], image_size=image_size, batch_size= batch_size
        )# root_ff, root_pf, image_size, batch_size
        self.train_loader = self.load_clip_dataloader(
            root_ff = train_targets[1], root_pf = train_targets[0], image_size=image_size, batch_size= batch_size
        )
        # self.val_full_loader = self.load_image_folder(
        #     val_targets[1], image_size, batch_size, num_workers
        # )
        # self.train_partial_loader = self.load_image_folder(
        #     train_targets[0], image_size, batch_size, num_workers
        # )
        # self.train_full_loader = self.load_image_folder(
        #     train_targets[1], image_size, batch_size, num_workers
        # )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    # def load_image_folder(self, path, image_size, batch_size, num_workers):
        # dataset = ImageFolder(
        #     path,
        #     transform=transforms.Compose([
        #         transforms.Resize(image_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5]*3, [0.5]*3),
        #     ])
        # )
        # return DataLoader(
        #     dataset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=num_workers,
        #     drop_last=False,
        # )
    def load_clip_dataloader(self, root_ff, root_pf, image_size, batch_size):
        return get_clip_dataloader(root_ff = root_ff, root_pf = root_pf, local_rank = 0, batch_size= batch_size)


    # --------------------------------------------------
    # Embedding extraction
    # --------------------------------------------------
    @torch.no_grad()
    def extract_embeddings(self, loader, backbone, max_embeddings):
        embeddings, labels = [], []

        backbone.eval()
        for i, (imgs, lbls) in enumerate(tqdm(loader, total=len(loader), desc="Extracting embeddings")):
            # imgs = imgs.cuda()
            emb = backbone(imgs.cuda())
            emb = F.normalize(emb)
            embeddings.append(emb.cpu())
            labels.append(lbls)
            if i >= max_embeddings // loader.batch_size:
                break

        return torch.cat(embeddings), torch.cat(labels)
    
    
    @torch.no_grad()
    def extract_embeddings_from_clip_dataloader(self, loader, backbone_full, backbone_partial, max_embeddings):
        pf_embeddings, ff_embeddings, labels = [], [], []

        backbone_full.eval()
        backbone_partial.eval()
        for i, (ff_imgs, pf_imgs, lbls) in enumerate(tqdm(loader, total=len(loader), desc="Extracting embeddings")):
            # imgs = imgs.cuda()
            pf_emb = backbone_partial(pf_imgs.cuda())
            pf_emb = F.normalize(pf_emb)
            pf_embeddings.append(pf_emb.cpu())

            ff_embs = backbone_full(ff_imgs.cuda())
            ff_embs = F.normalize(ff_embs)
            ff_embeddings.append(ff_embs.cpu())

            
            labels.append(lbls.cpu())
            if i >= max_embeddings // loader.batch_size:
                break

        return  torch.cat(ff_embeddings),torch.cat(pf_embeddings), torch.cat(labels)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    def tar_far_acc(self, Eq, Lq, Eg, Lg, thresholds, train_val_tag):
        S = Eq @ Eg.T  # (Nq, Ng)

        same = (Lq[:, None] == Lg[None, :])

        # Exclude diagonal only when it is meaningful (square matrix + aligned ordering)
        nq, ng = S.shape
        if nq == ng:
            eye = torch.eye(nq, dtype=torch.bool, device=S.device)
            pos_mask = same & ~eye      # same-id but different samples
            diag_mask = eye & same      # paired/diagonal (optional to inspect)
        else:
            pos_mask = same
            diag_mask = None

        neg_mask = ~same

        # hist
        same_offdiag_scores = S[pos_mask].detach().cpu().numpy()
        diff_scores         = S[neg_mask].detach().cpu().numpy()
        self.plot_histogram(same_offdiag_scores, diff_scores, train_val_tag)

        tar, far, balanced_acc = [], [], []

        for t in thresholds:
            accept = (S >= t)

            TP = (accept & pos_mask).sum().item()
            FP = (accept & neg_mask).sum().item()
            FN = ((~accept) & pos_mask).sum().item()
            TN = ((~accept) & neg_mask).sum().item()

            TAR = TP / (TP + FN + 1e-8)
            FAR = FP / (FP + TN + 1e-8)
            TNR = TN / (TN + FP + 1e-8)

            tar.append(TAR)
            far.append(FAR)
            balanced_acc.append(0.5 * (TAR + TNR))

        best_idx = int(torch.tensor(balanced_acc).argmax())
        return {
            "tar": tar,
            "far": far,
            "acc": balanced_acc,
            "best_acc": balanced_acc[best_idx],
            "best_threshold": thresholds[best_idx],
            # Optional debug:
            # "diag_mean": float(S[diag_mask].mean().item()) if diag_mask is not None else None
        }


    def rank1_accuracy(self, Eq, Lq, Eg, Lg):
        S = Eq @ Eg.T
        idx = S.argmax(dim=1)
        return (Lg[idx] == Lq).float().mean().item()


    def plot_histogram(self, same_scores, diff_scores, train_val_tag, seed=0):
        if train_val_tag != "val":
            return

        out = self.work_dir / f"ver_scores_distributions_epoch_{self.current_epoch}.png"

        same = np.asarray(same_scores).ravel()
        diff = np.asarray(diff_scores).ravel()

        #subsample diff so "Count" is comparable
        rng = np.random.default_rng(seed)
        if diff.size > same.size:
            diff = rng.choice(diff, size=same.size, replace=False)

        # Shared bins so the two hists align
        lo = min(same.min(), diff.min())
        hi = max(same.max(), diff.max())
        bins = np.linspace(lo, hi, 51)  # 50 bins

        fig, ax = plt.subplots()

        ax.hist(same, bins=bins, density=False,  label=f"Same ", alpha=0.7,)
        ax.hist(diff, bins=bins, density=False, label=f"Diff ", alpha=0.7,)

        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Same/Diff Pair Similarities")
        ax.legend()

        fig.savefig(out)
        plt.close(fig)
        print(f"Saved verification scores distribution plot to {out}")


    @torch.no_grad()
    def __call__(self, backbone_partial, backbone_full, global_step, epoch, max_embeddings=1000):
        self.current_epoch = epoch
        backbone_partial.eval()
        backbone_full.eval()
        thresholds = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 

        ## val set evaluation ##
        #extruct embeddings
        Eg, Eq, Lq = self.extract_embeddings_from_clip_dataloader(self.val_loader,backbone_full, backbone_partial, max_embeddings)
        Lg=Lq
        #verification metrics computation
        val_stats = self.tar_far_acc(Eq, Lq, Eg, Lg, thresholds, train_val_tag='val')
        val_rank1 = self.rank1_accuracy(Eq, Lq, Eg, Lg)

        ## train set evaluation ##
        #extruct embeddings
        Eg, Eq, Lq = self.extract_embeddings_from_clip_dataloader(self.train_loader,backbone_full, backbone_partial, max_embeddings)
        Lg=Lq
        #verification metrics computation
        train_stats = self.tar_far_acc(Eq, Lq, Eg, Lg, thresholds, train_val_tag='train')
        train_rank1 = self.rank1_accuracy(Eq, Lq, Eg, Lg)

        #log and summary write results
        logging.info(
            f"[CLIP][{global_step}] train verification test "
            f"Train BestAcc={train_stats['best_acc']:.4f} "
            f"Train Rank1={train_rank1:.4f} "
            f"Train Thr={train_stats['best_threshold']:.2f}"
        )

        logging.info(
            f"[CLIP][{global_step}][VAL] "
            f"Val BestAcc={val_stats['best_acc']:.4f} "
            f"Val Rank1={val_rank1:.4f} "
            f"Val Thr={val_stats['best_threshold']:.2f}"
        )

        if self.summary_writer:
            self.summary_writer.add_scalars(
                "CLIP/BestAccuracy",
                {
                    "Train": train_stats["best_acc"],
                    "Val":   val_stats["best_acc"],
                },
                epoch,
            )

            self.summary_writer.add_scalars(
                "CLIP/Rank1",
                {
                    "Train": train_rank1,
                    "Val":   val_rank1,
                },
                epoch,
            )
        backbone_partial.train()
        # backbone_full.train() this is the teacher model, we don't train it
        return val_stats["best_acc"], val_rank1
