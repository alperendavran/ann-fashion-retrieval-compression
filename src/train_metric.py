"""
Part B: train retrieval embeddings from the selected Part A backbone.

The default is triplet loss because it maps directly to retrieval: an anchor
should be closer to a same-class item than to a different-class item.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from config import METRIC_BATCH_SIZE, METRIC_EVAL_SIZE, METRIC_TRAIN_SIZE, SEED
from data import (
    get_metric_dataloader,
    get_metric_label_dataloader,
    get_retrieval_eval_dataloader,
    get_retrieval_val_dataloader,
)
from losses import batch_hard_triplet_loss, contrastive_loss, triplet_loss
from models import build_model
from utils import (
    append_csv,
    collect_embeddings,
    count_parameters,
    ensure_dir,
    estimate_flops,
    get_device,
    load_checkpoint,
    load_matching_weights,
    retrieval_metrics,
    save_checkpoint,
    set_seed,
    write_json,
)

ALLOWED_LOSSES = ("triplet", "contrastive")


def resolve_part_a_checkpoint(path: str | None) -> str:
    if path:
        return path
    selected_path = Path("outputs/part_a/selected_backbone.json")
    if not selected_path.exists():
        raise FileNotFoundError("Run Part A first or pass --part-a-checkpoint.")
    return str(json.loads(selected_path.read_text(encoding="utf-8"))["checkpoint"])


def build_embedding_model(part_a_checkpoint: str, embedding_dim: int, device: torch.device) -> tuple[nn.Module, str]:
    checkpoint = load_checkpoint(part_a_checkpoint, map_location=device)
    model_name = checkpoint["metadata"]["model_name"]
    model = build_model(model_name, embedding_dim=embedding_dim).to(device)
    matched_keys = load_matching_weights(model, checkpoint["model_state"])
    print(f"Loaded {len(matched_keys)} tensors from Part A checkpoint into {model_name}.")
    return model, model_name


def batch_loss(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    loss_name: str,
    device: torch.device,
    margin: float,
) -> tuple[torch.Tensor, int]:
    if loss_name == "triplet":
        anchor, positive, negative = [item.to(device) for item in batch]
        anchor_embedding = F.normalize(model(anchor), p=2, dim=1)
        positive_embedding = F.normalize(model(positive), p=2, dim=1)
        negative_embedding = F.normalize(model(negative), p=2, dim=1)
        return triplet_loss(anchor_embedding, positive_embedding, negative_embedding, margin=margin), anchor.size(0)
    if loss_name == "contrastive":
        image_a, image_b, same_class = batch
        image_a = image_a.to(device)
        image_b = image_b.to(device)
        same_class = same_class.to(device)
        embedding_a = F.normalize(model(image_a), p=2, dim=1)
        embedding_b = F.normalize(model(image_b), p=2, dim=1)
        return contrastive_loss(embedding_a, embedding_b, same_class, margin=margin), image_a.size(0)
    raise ValueError(f"Unsupported retrieval loss: {loss_name}")


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_name: str,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        optimizer.zero_grad()
        loss, batch_size = batch_loss(model, batch, loss_name, device, margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / total_examples


def train_epoch_mining(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float,
) -> float:
    """Triplet training with online batch-hard mining over (image, label) batches."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        embeddings = F.normalize(model(images), p=2, dim=1)
        loss = batch_hard_triplet_loss(embeddings, labels, margin=margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_examples += images.size(0)
    return total_loss / total_examples


@torch.no_grad()
def evaluate_retrieval(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    embeddings, labels, _images = collect_embeddings(model, dataloader, device, normalize=True)
    return retrieval_metrics(embeddings, labels, ks=(1, 5, 10))


def train_one_configuration(
    loss_name: str,
    part_a_checkpoint: str,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    margin: float,
    output_dir: Path,
    metric_train_size: int,
    metric_eval_size: int,
    device: torch.device,
    metric_augmentation: bool,
) -> dict[str, float | str | int | bool]:
    set_seed(SEED)
    model, backbone = build_embedding_model(part_a_checkpoint, embedding_dim, device)
    use_mining = loss_name == "triplet"
    if use_mining:
        train_loader = get_metric_label_dataloader(
            batch_size=batch_size,
            subset_size=metric_train_size,
            augment=metric_augmentation,
        )
    else:
        train_loader = get_metric_dataloader(
            loss_name=loss_name,
            batch_size=batch_size,
            subset_size=metric_train_size,
            augment=metric_augmentation,
        )
    eval_loader = get_retrieval_eval_dataloader(batch_size=256, eval_size=metric_eval_size)
    val_loader = get_retrieval_val_dataloader(batch_size=256, eval_size=metric_eval_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Checkpoint selection uses the disjoint validation split; the reported eval
    # set is only measured once, on the finally selected checkpoint.
    baseline_metrics = evaluate_retrieval(model, eval_loader, device)
    best_state = deepcopy(model.state_dict())
    best_val_recall = evaluate_retrieval(model, val_loader, device)["recall@1"]
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        if use_mining:
            loss = train_epoch_mining(model, train_loader, optimizer, device, margin)
        else:
            loss = train_epoch(model, train_loader, loss_name, optimizer, device, margin)
        val_metrics = evaluate_retrieval(model, val_loader, device)
        scheduler.step(val_metrics["recall@1"])
        print(
            f"{backbone} {loss_name} epoch={epoch:02d} "
            f"loss={loss:.4f} val_recall@1={val_metrics['recall@1']:.4f} val_recall@5={val_metrics['recall@5']:.4f}"
        )
        if val_metrics["recall@1"] >= best_val_recall:
            best_state = deepcopy(model.state_dict())
            best_val_recall = val_metrics["recall@1"]
            best_epoch = epoch

    model.load_state_dict(best_state)
    best_metrics = evaluate_retrieval(model, eval_loader, device)
    checkpoint_path = output_dir / f"retrieval_{backbone}_{loss_name}_best.pt"
    metadata = {
        "stage": "part_b_retrieval",
        "backbone": backbone,
        "loss_name": loss_name,
        "mining": "batch_hard_soft_margin" if use_mining else "none",
        "part_a_checkpoint": part_a_checkpoint,
        "embedding_dim": embedding_dim,
        "learning_rate": learning_rate,
        "margin": margin,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "metric_augmentation": metric_augmentation,
        "selection": "disjoint_val_split",
        "val_selection_recall@1": best_val_recall,
        "baseline_metrics": baseline_metrics,
        "retrieval_metrics": best_metrics,
        "parameters": count_parameters(model),
        "flops": estimate_flops(model),
    }
    save_checkpoint(checkpoint_path, model, metadata)

    row = {
        "backbone": backbone,
        "loss_name": loss_name,
        "embedding_dim": embedding_dim,
        "best_epoch": best_epoch,
        "metric_augmentation": metric_augmentation,
        "recall@1_baseline": baseline_metrics["recall@1"],
        "recall@5_baseline": baseline_metrics["recall@5"],
        "recall@10_baseline": baseline_metrics["recall@10"],
        "mAP_baseline": baseline_metrics["mAP"],
        "recall@1": best_metrics["recall@1"],
        "recall@5": best_metrics["recall@5"],
        "recall@10": best_metrics["recall@10"],
        "mAP": best_metrics["mAP"],
        "parameters": metadata["parameters"],
        "flops": metadata["flops"],
        "checkpoint": str(checkpoint_path),
    }
    append_csv(output_dir / "retrieval_results.csv", row)
    return row


def select_retrieval_model(results: list[dict[str, float | str | int | bool]]) -> dict[str, float | str | int | bool]:
    return max(results, key=lambda row: (float(row["recall@1"]), float(row["recall@5"]), -int(row["flops"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-a-checkpoint", default=None)
    parser.add_argument("--losses", nargs="+", default=["triplet"], choices=list(ALLOWED_LOSSES))
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=METRIC_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/part_b")
    parser.add_argument("--metric-train-size", type=int, default=METRIC_TRAIN_SIZE)
    parser.add_argument("--metric-eval-size", type=int, default=METRIC_EVAL_SIZE)
    parser.add_argument("--no-metric-augmentation", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.metric_train_size = min(args.metric_train_size, 512)
        args.metric_eval_size = min(args.metric_eval_size, 256)

    output_dir = ensure_dir(args.output_dir)
    results_csv = output_dir / "retrieval_results.csv"
    if results_csv.exists():
        results_csv.unlink()

    device = get_device()
    part_a_checkpoint = resolve_part_a_checkpoint(args.part_a_checkpoint)
    print(f"Using device: {device}")
    print(f"Part A checkpoint: {part_a_checkpoint}")

    results = [
        train_one_configuration(
            loss_name=loss_name,
            part_a_checkpoint=part_a_checkpoint,
            embedding_dim=args.embedding_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            margin=args.margin,
            output_dir=output_dir,
            metric_train_size=args.metric_train_size,
            metric_eval_size=args.metric_eval_size,
            device=device,
            metric_augmentation=not args.no_metric_augmentation,
        )
        for loss_name in args.losses
    ]

    selected = select_retrieval_model(results)
    write_json(output_dir / "retrieval_results.json", results)
    write_json(output_dir / "selected_retrieval_model.json", selected)
    print(f"Selected retrieval model: {selected['checkpoint']}")


if __name__ == "__main__":
    main()
