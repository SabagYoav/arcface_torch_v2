import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from dataset import crop_roi, get_transform
from eval.benchmarks.lfw_bin import decode_lfw_bin, compute_or_load_eye_centers, pair_folds


class ClipVerification(object):
    """LFW-based cross-domain verification, run periodically during training.

    Three modes, computed from the SAME 6000 LFW pairs:
      teacher_full_vs_full      : R50 teacher on full faces  (standard LFW upper bound)
      cross_partial_vs_full     : student on partial-face probe vs teacher on full-face
                                   gallery (the actual deployment-relevant metric)
      student_partial_vs_partial: student on partial faces only (same-domain)

    Full-face and partial-face LFW crops are precomputed once per ratio (LFW is
    small — 12k images — so no on-the-fly-per-batch cost like the training set).
    Partial crops use the same canonical crop_roi() as on-the-fly training data,
    so train/val ROI geometry is consistent.

    Optional (off by default, backward-compatible — only used by the architecture-
    comparison sweep, not the main ROI sweep): pass `glint_val_root` +
    `glint_val_metadata` to ALSO measure cross_partial_vs_full accuracy against a
    held-out glint360k identity subset (open-set, N-vs-N gallery/probe, unlike
    LFW's fixed pairing), scored at the same ratio being trained.
    """

    N_FOLDS = 10
    THRESHOLDS = np.linspace(-1, 1, 4001)
    OPEN_SET_THRESHOLDS = np.linspace(-1, 1, 201)  # coarser: NxN sweep is O(N^2) per threshold

    def __init__(
        self,
        ratio,
        summary_writer=None,
        wandb_logger=None,
        work_dir="./",
        batch_size=128,
        width_ratio=1.0,
        glint_val_root=None,
        glint_val_metadata=None,
        glint_val_max_images=2000,
        glint_val_seed=42,
    ):
        self.ratio = ratio
        self.width_ratio = width_ratio
        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger
        self.work_dir = Path(work_dir)
        self.batch_size = batch_size
        self.current_epoch = -1

        images, issame = decode_lfw_bin()
        self.issame = np.asarray(issame, dtype=bool)
        self.folds = pair_folds(len(issame), self.N_FOLDS)

        app = self._build_detector()
        centers = compute_or_load_eye_centers(images, app)

        transform = get_transform(augmentations=False)
        full_tensors, partial_tensors = [], []
        for bgr, meta in zip(images, centers):
            partial_bgr = crop_roi(bgr, ratio, meta["center_y"], width_ratio)
            full_tensors.append(transform(self._to_pil(bgr)))
            partial_tensors.append(transform(self._to_pil(partial_bgr)))
        self.full_tensor = torch.stack(full_tensors)
        self.partial_tensor = torch.stack(partial_tensors)

        self.glint_val_enabled = glint_val_root is not None and glint_val_metadata is not None
        if self.glint_val_enabled:
            self._init_glint_val(
                Path(glint_val_root), glint_val_metadata, ratio, width_ratio,
                glint_val_max_images, glint_val_seed, transform,
            )

    def _init_glint_val(self, val_root, metadata, ratio, width_ratio, max_images, seed, transform):
        import cv2

        detected_items = [k for k, v in metadata.items() if v.get("detected", False)]
        rng = np.random.default_rng(seed)
        rng.shuffle(detected_items)
        sample_keys = detected_items[:max_images]

        full_tensors, partial_tensors, labels = [], [], []
        id_to_label = {}
        for key in sample_keys:
            id_name, filename = key.split("/", 1)
            bgr = cv2.imread(str(val_root / id_name / filename))
            if bgr is None:
                continue
            partial_bgr = crop_roi(bgr, ratio, metadata[key]["center_y"], width_ratio)
            full_tensors.append(transform(self._to_pil(bgr)))
            partial_tensors.append(transform(self._to_pil(partial_bgr)))
            labels.append(id_to_label.setdefault(id_name, len(id_to_label)))

        self.glint_full_tensor = torch.stack(full_tensors)
        self.glint_partial_tensor = torch.stack(partial_tensors)
        self.glint_labels = torch.tensor(labels)
        print(
            f"ClipVerification: glint-val open-set check ready "
            f"({len(labels)} images, {len(id_to_label)} identities)"
        )

    @staticmethod
    def _build_detector():
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l", allowed_modules=["detection"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(160, 160), det_thresh=0.5)
        return app

    @staticmethod
    def _to_pil(bgr):
        import cv2
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # --------------------------------------------------
    # Embedding extraction
    # --------------------------------------------------
    @torch.no_grad()
    def _embed(self, backbone, tensor):
        backbone.eval()
        out = []
        for i in range(0, tensor.shape[0], self.batch_size):
            batch = tensor[i:i + self.batch_size].cuda()
            emb = F.normalize(backbone(batch))
            out.append(emb.cpu())
        return torch.cat(out)

    # --------------------------------------------------
    # Metrics (10-fold acc, global best-threshold acc, AUC, TAR@FAR)
    # --------------------------------------------------
    def _metrics(self, scores):
        scores = np.asarray(scores)
        labels = self.issame
        folds = self.folds

        accs = []
        for f in range(self.N_FOLDS):
            tr, te = folds != f, folds == f
            fold_accs = [((scores[tr] >= t) == labels[tr]).mean() for t in self.THRESHOLDS]
            best_t = self.THRESHOLDS[int(np.argmax(fold_accs))]
            accs.append(((scores[te] >= best_t) == labels[te]).mean())
        accs = np.array(accs)

        acc_all = np.array([((scores >= t) == labels).mean() for t in self.THRESHOLDS])
        gi = int(np.argmax(acc_all))

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        def tar_at(far_target):
            idx = np.where(fpr <= far_target)[0]
            return float(tpr[idx[-1]]) if len(idx) else 0.0

        return {
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std()),
            "best_acc": float(acc_all[gi]),
            "best_thr": float(self.THRESHOLDS[gi]),
            "auc": float(roc_auc),
            "tar@1e-2": tar_at(1e-2),
            "tar@1e-3": tar_at(1e-3),
            "same_scores": scores[labels],
            "diff_scores": scores[~labels],
        }

    # --------------------------------------------------
    # Open-set metrics for the glint-val check (N-vs-N gallery/probe, no fixed
    # pairing like LFW — positive = same identity, off-diagonal).
    # --------------------------------------------------
    def _open_set_metrics(self, Ep, Ef, labels):
        S = (Ep @ Ef.T).numpy()
        labels = labels.numpy()
        N = S.shape[0]
        same = labels[:, None] == labels[None, :]
        eye = np.eye(N, dtype=bool)
        pos_mask = same & ~eye
        neg_mask = ~same

        pos_scores = S[pos_mask]
        neg_scores = S[neg_mask]

        best_acc, best_thr = 0.0, 0.0
        for t in self.OPEN_SET_THRESHOLDS:
            tar = (pos_scores >= t).mean()
            tnr = (neg_scores < t).mean()
            acc = 0.5 * (tar + tnr)
            if acc > best_acc:
                best_acc, best_thr = float(acc), float(t)

        all_scores = np.concatenate([pos_scores, neg_scores])
        all_labels = np.concatenate([np.ones_like(pos_scores, dtype=bool), np.zeros_like(neg_scores, dtype=bool)])
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)

        return {
            "best_acc": best_acc,
            "best_thr": best_thr,
            "auc": float(roc_auc),
            "same_scores": pos_scores,
            "diff_scores": neg_scores,
        }

    def plot_histogram(self, stats, tag):
        out = self.work_dir / f"ver_scores_distributions_{tag}_epoch_{self.current_epoch}.png"
        same, diff = stats["same_scores"], stats["diff_scores"]

        rng = np.random.default_rng(0)
        if diff.size > same.size:
            diff = rng.choice(diff, size=same.size, replace=False)

        lo = min(same.min(), diff.min())
        hi = max(same.max(), diff.max())
        bins = np.linspace(lo, hi, 51)

        fig, ax = plt.subplots()
        ax.hist(same, bins=bins, density=False, label="Same", alpha=0.7)
        ax.hist(diff, bins=bins, density=False, label="Diff", alpha=0.7)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"LFW {tag}: Same/Diff Pair Similarities")
        ax.legend()
        fig.savefig(out)
        plt.close(fig)

    @torch.no_grad()
    def __call__(self, backbone_partial, backbone_full, global_step, epoch, max_embeddings=None):
        self.current_epoch = epoch
        backbone_partial.eval()
        backbone_full.eval()

        Ef = self._embed(backbone_full, self.full_tensor)      # teacher, full-face
        Ep = self._embed(backbone_partial, self.partial_tensor)  # student, partial-face

        a, b = np.arange(0, Ef.shape[0], 2), np.arange(1, Ef.shape[0], 2)
        S_cross = 0.5 * ((Ep[a] * Ef[b]).sum(dim=1).numpy() + (Ep[b] * Ef[a]).sum(dim=1).numpy())

        cross_stats = self._metrics(S_cross)
        self.plot_histogram(cross_stats, "cross_partial_vs_full")

        logging.info(
            f"[LFW][{global_step}] cross_partial_vs_full "
            f"Acc={cross_stats['acc_mean']:.4f}±{cross_stats['acc_std']:.4f} "
            f"AUC={cross_stats['auc']:.4f} BestAcc={cross_stats['best_acc']:.4f} "
            f"TAR@1e-2={cross_stats['tar@1e-2']:.4f} TAR@1e-3={cross_stats['tar@1e-3']:.4f}"
        )

        if self.summary_writer:
            self.summary_writer.add_scalars(
                "LFW/BestAccuracy",
                {"cross_partial_vs_full": cross_stats["best_acc"]},
                epoch,
            )

        if self.glint_val_enabled:
            Ef_g = self._embed(backbone_full, self.glint_full_tensor)
            Ep_g = self._embed(backbone_partial, self.glint_partial_tensor)
            glint_stats = self._open_set_metrics(Ep_g, Ef_g, self.glint_labels)
            self.plot_histogram(glint_stats, "glint_val_cross_partial_vs_full")

            logging.info(
                f"[GlintVal][{global_step}] cross_partial_vs_full "
                f"BestAcc={glint_stats['best_acc']:.4f} AUC={glint_stats['auc']:.4f}"
            )
            if self.summary_writer:
                self.summary_writer.add_scalars(
                    "GlintVal/BestAccuracy",
                    {"cross_partial_vs_full": glint_stats["best_acc"]},
                    epoch,
                )

        backbone_partial.train()
        # backbone_full.train() this is the teacher model, we don't train it

        return cross_stats["best_acc"], cross_stats["acc_mean"]
