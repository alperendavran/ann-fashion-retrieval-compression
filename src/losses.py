"""
Losses used by the compact project pipeline.

Part B uses one of the two losses allowed in the brief: contrastive loss or
triplet loss. Part C distills retrieval geometry with RKD while keeping the
same metric-learning objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
    same_class: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Hadsell, Chopra, LeCun contrastive loss for positive/negative pairs."""
    distances = F.pairwise_distance(embedding_a, embedding_b, p=2)
    positive = same_class * distances.pow(2)
    negative = (1.0 - same_class) * F.relu(margin - distances).pow(2)
    return (positive + negative).mean()


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """FaceNet-style triplet margin loss for anchor, positive, and negative images."""
    return F.triplet_margin_loss(anchor, positive, negative, margin=margin, p=2)


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    soft_margin: bool = True,
) -> torch.Tensor:
    """
    Online batch-hard triplet loss (Hermans et al., 2017).

    Random triplets are dominated by easy negatives that already satisfy the
    margin, so they produce almost no gradient. For each anchor this selects the
    hardest positive (farthest same-class sample) and the hardest negative
    (closest different-class sample) inside the batch, which keeps the gradient
    focused on informative triplets.
    """
    distances = torch.cdist(embeddings, embeddings, p=2)
    labels = labels.view(-1, 1)
    same_label = labels == labels.t()
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=embeddings.device)

    positive_mask = same_label & ~eye
    negative_mask = ~same_label

    # Anchors that have at least one positive and one negative in the batch.
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if valid.sum() == 0:
        return embeddings.sum() * 0.0

    hardest_positive = (distances * positive_mask).max(dim=1).values
    masked_negative = distances + (~negative_mask) * 1e6
    hardest_negative = masked_negative.min(dim=1).values

    pos = hardest_positive[valid]
    neg = hardest_negative[valid]
    if soft_margin:
        return F.softplus(pos - neg).mean()
    return F.relu(pos - neg + margin).mean()


def _pairwise_distance_matrix(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    squared_norm = embeddings.pow(2).sum(dim=1)
    distances = squared_norm.unsqueeze(0) + squared_norm.unsqueeze(1) - 2 * (embeddings @ embeddings.t())
    distances = distances.clamp(min=eps).sqrt()
    distances = distances.clone()
    distances[range(len(embeddings)), range(len(embeddings))] = 0.0
    return distances


def rkd_distance_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    """Distance-wise Relational Knowledge Distillation."""
    with torch.no_grad():
        teacher_distances = _pairwise_distance_matrix(teacher_embeddings)
        teacher_distances = teacher_distances / teacher_distances[teacher_distances > 0].mean()

    student_distances = _pairwise_distance_matrix(student_embeddings)
    student_distances = student_distances / student_distances[student_distances > 0].mean()
    return F.smooth_l1_loss(student_distances, teacher_distances)


def rkd_angle_loss(student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
    """Angle-wise Relational Knowledge Distillation."""
    with torch.no_grad():
        teacher_diff = teacher_embeddings.unsqueeze(0) - teacher_embeddings.unsqueeze(1)
        teacher_diff = F.normalize(teacher_diff, p=2, dim=2)
        teacher_angle = torch.bmm(teacher_diff, teacher_diff.transpose(1, 2)).view(-1)

    student_diff = student_embeddings.unsqueeze(0) - student_embeddings.unsqueeze(1)
    student_diff = F.normalize(student_diff, p=2, dim=2)
    student_angle = torch.bmm(student_diff, student_diff.transpose(1, 2)).view(-1)
    return F.smooth_l1_loss(student_angle, teacher_angle)


def rkd_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    distance_weight: float = 1.0,
    angle_weight: float = 2.0,
) -> torch.Tensor:
    """RKD distance + angle loss; preserves retrieval geometry, not class logits."""
    return distance_weight * rkd_distance_loss(student_embeddings, teacher_embeddings) + angle_weight * rkd_angle_loss(
        student_embeddings,
        teacher_embeddings,
    )
