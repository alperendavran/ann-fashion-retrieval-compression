"""
Common utilities for the FashionMNIST retrieval-compression project.

The helpers in this file keep the experiment scripts small and consistent:
seed handling, metric computation, checkpoint I/O, parameter counts, and a
lightweight FLOP estimator for the fixed CNN scaffold.
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from config import FASHION_MNIST_MEAN, FASHION_MNIST_STD


FASHION_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def count_nonzero_parameters(model: nn.Module) -> int:
    return sum(int(torch.count_nonzero(p.detach()).item()) for p in model.parameters())


def macro_f1_from_counts(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 10) -> float:
    predictions = predictions.cpu()
    targets = targets.cpu()
    f1_values = []
    for class_id in range(num_classes):
        pred_is_class = predictions == class_id
        target_is_class = targets == class_id
        tp = torch.logical_and(pred_is_class, target_is_class).sum().item()
        fp = torch.logical_and(pred_is_class, ~target_is_class).sum().item()
        fn = torch.logical_and(~pred_is_class, target_is_class).sum().item()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_values.append(f1)
    return float(sum(f1_values) / len(f1_values))


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 10) -> dict[str, float]:
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1_from_counts(predictions, targets, num_classes=num_classes),
    }


def forward_embeddings(model: nn.Module, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    embeddings = model(images)
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def denormalize_fashion_images(images: torch.Tensor) -> torch.Tensor:
    """Invert the shared FashionMNIST channel normalization for plotting."""
    mean = torch.tensor(FASHION_MNIST_MEAN, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(FASHION_MNIST_STD, dtype=images.dtype, device=images.device).view(1, -1, 1, 1)
    return images * std + mean


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    all_embeddings = []
    all_labels = []
    all_images = []
    for images, labels in dataloader:
        images = images.to(device)
        embeddings = forward_embeddings(model, images, normalize=normalize)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
        all_images.append(images.cpu())
    return torch.cat(all_embeddings), torch.cat(all_labels), torch.cat(all_images)


def _nearest_indices(embeddings: torch.Tensor, max_k: int) -> torch.Tensor:
    if embeddings.size(0) <= max_k:
        raise ValueError("Need more evaluation examples than max_k.")
    distances = torch.cdist(embeddings, embeddings, p=2)
    distances.fill_diagonal_(float("inf"))
    return distances.topk(max_k, largest=False).indices


def recall_at_k(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    nearest = _nearest_indices(embeddings, k)
    matches = labels[nearest] == labels.unsqueeze(1)
    return float(matches.any(dim=1).float().mean().item())


def mean_average_precision(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    max_k: int | None = None,
) -> float:
    """
    Mean average precision for same-class retrieval.

    The query item itself is excluded by the infinite diagonal in the distance
    matrix. `max_k=None` evaluates the full ranked list; a positive `max_k`
    gives mAP@k.
    """
    max_neighbors = embeddings.size(0) - 1
    if max_k is None:
        max_k = max_neighbors
    max_k = min(max_k, max_neighbors)
    nearest = _nearest_indices(embeddings, max_k)
    matches = (labels[nearest] == labels.unsqueeze(1)).float()
    cumulative_hits = matches.cumsum(dim=1)
    ranks = torch.arange(1, max_k + 1, dtype=torch.float32).unsqueeze(0)
    precision_at_rank = cumulative_hits / ranks
    positives_per_query = (labels.unsqueeze(0) == labels.unsqueeze(1)).sum(dim=1).float() - 1.0
    normalizer = torch.minimum(
        positives_per_query.clamp(min=1.0),
        torch.tensor(float(max_k), dtype=torch.float32, device=labels.device),
    )
    average_precision = (precision_at_rank * matches).sum(dim=1) / normalizer
    valid = positives_per_query > 0
    if valid.sum() == 0:
        return 0.0
    return float(average_precision[valid].mean().item())


def retrieval_metrics(embeddings: torch.Tensor, labels: torch.Tensor, ks: Iterable[int] = (1, 5, 10)) -> dict[str, float]:
    ks = tuple(sorted(set(int(k) for k in ks)))
    max_k = max(ks)
    nearest = _nearest_indices(embeddings, max_k)
    metrics = {}
    for k in ks:
        matches = labels[nearest[:, :k]] == labels.unsqueeze(1)
        metrics[f"recall@{k}"] = float(matches.any(dim=1).float().mean().item())
    metrics["mAP"] = mean_average_precision(embeddings, labels)
    return metrics


def per_class_recall_at_k(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 1) -> dict[str, float]:
    nearest = _nearest_indices(embeddings, k)
    matches = labels[nearest] == labels.unsqueeze(1)
    hit = matches.any(dim=1).float()
    scores = {}
    for class_id, class_name in enumerate(FASHION_CLASSES):
        mask = labels == class_id
        scores[class_name] = float(hit[mask].mean().item()) if mask.any() else 0.0
    return scores


def retrieval_confusion_matrix(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Top-1 retrieval confusion matrix.

    Rows are true query classes and columns are the class retrieved by the
    nearest neighbor. This exposes which FashionMNIST classes are mixed up in
    the embedding space, which Recall@1 alone cannot show.
    """
    nearest = _nearest_indices(embeddings, 1).squeeze(1)
    retrieved = labels[nearest]
    matrix = torch.zeros((len(FASHION_CLASSES), len(FASHION_CLASSES)), dtype=torch.int64)
    for true_label, retrieved_label in zip(labels.cpu(), retrieved.cpu()):
        matrix[int(true_label), int(retrieved_label)] += 1
    return matrix


def embedding_geometry_diagnostics(embeddings: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """
    Alignment/uniformity and intra/inter distance diagnostics.

    Alignment and uniformity follow Wang & Isola (ICML 2020): alignment is the
    expected squared distance of positive pairs, and uniformity is
    log E exp(-2||x-y||^2) over all pairs. Lower alignment means tighter
    same-class neighborhoods; lower uniformity means points are more evenly
    spread on the hypersphere.
    """
    labels = labels.cpu()
    embeddings = F.normalize(embeddings.cpu(), p=2, dim=1)
    squared_distances = torch.cdist(embeddings, embeddings, p=2).pow(2)
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    identity = torch.eye(labels.size(0), dtype=torch.bool)
    positive_mask = same_class & ~identity
    negative_mask = ~same_class
    upper_triangle = torch.triu(torch.ones_like(squared_distances, dtype=torch.bool), diagonal=1)

    alignment = float(squared_distances[positive_mask].mean().item()) if positive_mask.any() else 0.0
    uniformity = float(torch.log(torch.exp(-2.0 * squared_distances[upper_triangle]).mean()).item())
    intra_distance = float(torch.sqrt(squared_distances[positive_mask]).mean().item()) if positive_mask.any() else 0.0
    inter_distance = float(torch.sqrt(squared_distances[negative_mask]).mean().item()) if negative_mask.any() else 0.0
    return {
        "alignment": alignment,
        "uniformity": uniformity,
        "intra_class_distance": intra_distance,
        "inter_class_distance": inter_distance,
        "intra_inter_distance_ratio": intra_distance / inter_distance if inter_distance else 0.0,
    }


def save_checkpoint(path: str | os.PathLike[str], model: nn.Module, metadata: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | os.PathLike[str], map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def load_matching_weights(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    own_state = model.state_dict()
    matched = {
        key: value
        for key, value in state_dict.items()
        if key in own_state and own_state[key].shape == value.shape
    }
    own_state.update(matched)
    model.load_state_dict(own_state)
    return sorted(matched)


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_csv(path: str | os.PathLike[str], row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def estimate_flops(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 1, 28, 28),
    multiply_adds: int = 2,
) -> int:
    """
    Estimate dense Conv2d/Linear FLOPs for one forward pass.

    This follows the same comparison goal as the exercise FLOP analysis: it is
    a consistent model-to-model estimate, not a hardware latency benchmark.
    """
    hooks = []
    total_flops = 0

    def conv_hook(module: nn.Conv2d, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal total_flops
        batch_size, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        in_channels_per_group = module.in_channels // module.groups
        ops_per_output = kernel_h * kernel_w * in_channels_per_group * multiply_adds
        if module.bias is not None:
            ops_per_output += 1
        total_flops += int(batch_size * out_channels * out_h * out_w * ops_per_output)

    def linear_hook(module: nn.Linear, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal total_flops
        batch_size = output.shape[0] if output.dim() > 1 else 1
        ops_per_output = module.in_features * multiply_adds
        if module.bias is not None:
            ops_per_output += 1
        total_flops += int(batch_size * module.out_features * ops_per_output)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy = torch.zeros(input_shape, device=device)
    with torch.no_grad():
        model(dummy)
    model.train(training)
    for hook in hooks:
        hook.remove()
    return total_flops
