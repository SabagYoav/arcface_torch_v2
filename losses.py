import torch
import math


class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s        

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


class BatchAllTripletLoss(torch.nn.Module):
    """Batch-all triplet loss using an NxN pairwise distance matrix.

    Enumerates all valid (anchor, positive, negative) triplets in the batch
    and averages the margin losses over triplets with positive loss.

    Uses cosine distance: d = 1 - cos_sim.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # NxN cosine similarity -> distance
        dist_matrix = 1.0 - torch.mm(embeddings, embeddings.t())  # [N, N]

        N = labels.size(0)
        same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1))  # [N, N]

        # Positive mask: same class, exclude self
        pos_mask = same_class.clone()
        pos_mask.fill_diagonal_(False)

        # Negative mask: different class
        neg_mask = ~same_class

        # Per-anchor mean positive distance and mean negative distance
        num_pos = pos_mask.sum(dim=1).float().clamp(min=1.0)  # [N]
        num_neg = neg_mask.sum(dim=1).float().clamp(min=1.0)  # [N]

        mean_d_pos = (dist_matrix * pos_mask.float()).sum(dim=1) / num_pos  # [N]
        mean_d_neg = (dist_matrix * neg_mask.float()).sum(dim=1) / num_neg  # [N]

        # Triplet loss per anchor (balanced: one pos term, one neg term per anchor)
        per_anchor_loss = torch.nn.functional.relu(mean_d_pos - mean_d_neg + self.margin)

        # Only average over anchors that have at least one positive
        valid = pos_mask.any(dim=1)
        if valid.sum() == 0:
            return (embeddings * 0).sum()  # zero loss, keeps grad graph

        return per_anchor_loss[valid].mean()
