"""
Part C: compress the Part B retrieval model.

Two required compression methods are evaluated:
- global unstructured L1 pruning with retrieval fine-tuning,
- RKD distillation into the fixed CompactSeparableCNN student.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from config import COMPRESSION_STUDENT_MODEL, METRIC_BATCH_SIZE, METRIC_EVAL_SIZE, METRIC_TRAIN_SIZE, SEED
from data import get_metric_dataloader, get_retrieval_eval_dataloader, get_retrieval_val_dataloader
from losses import contrastive_loss, rkd_loss, triplet_loss
from models import build_model
from utils import (
    append_csv,
    collect_embeddings,
    count_nonzero_parameters,
    count_parameters,
    ensure_dir,
    estimate_flops,
    get_device,
    load_checkpoint,
    retrieval_metrics,
    save_checkpoint,
    set_seed,
    write_json,
)


def resolve_retrieval_checkpoint(path: str | None) -> str:
    if path:
        return path
    selected_path = Path("outputs/part_b/selected_retrieval_model.json")
    if not selected_path.exists():
        raise FileNotFoundError("Run Part B first or pass --retrieval-checkpoint.")
    return str(json.loads(selected_path.read_text(encoding="utf-8"))["checkpoint"])


def load_retrieval_model(path: str, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = load_checkpoint(path, map_location=device)
    metadata = checkpoint["metadata"]
    model = build_model(metadata["backbone"], embedding_dim=metadata["embedding_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, metadata


def prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    return [(module, "weight") for module in model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))]


def apply_global_l1_pruning(model: nn.Module, amount: float) -> None:
    prune.global_unstructured(prunable_modules(model), pruning_method=prune.L1Unstructured, amount=amount)


def remove_pruning_reparametrization(model: nn.Module) -> None:
    for module, name in prunable_modules(model):
        if hasattr(module, f"{name}_orig"):
            prune.remove(module, name)


def masked_weight_sparsity(model: nn.Module) -> float:
    total = 0
    nonzero = 0
    for module, name in prunable_modules(model):
        weight = getattr(module, name)
        total += weight.numel()
        nonzero += int(torch.count_nonzero(weight).item())
    return 1.0 - nonzero / total if total else 0.0


def retrieval_batch_loss(
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
    raise ValueError("Part C expects a triplet or contrastive Part B teacher.")


def train_retrieval_epoch(
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
        loss, batch_size = retrieval_batch_loss(model, batch, loss_name, device, margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / total_examples


@torch.no_grad()
def evaluate_retrieval(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    embeddings, labels, _images = collect_embeddings(model, dataloader, device, normalize=True)
    return retrieval_metrics(embeddings, labels, ks=(1, 5, 10))


def train_pruned_model(
    teacher_path: str,
    teacher_metadata: dict,
    amount: float,
    epochs: int,
    learning_rate: float,
    margin: float,
    output_dir: Path,
    metric_train_size: int,
    metric_eval_size: int,
    device: torch.device,
    metric_augmentation: bool,
) -> dict[str, float | str | int | bool]:
    model, _ = load_retrieval_model(teacher_path, device)
    loss_name = teacher_metadata["loss_name"]
    train_loader = get_metric_dataloader(
        loss_name=loss_name,
        batch_size=METRIC_BATCH_SIZE,
        subset_size=metric_train_size,
        augment=metric_augmentation,
    )
    eval_loader = get_retrieval_eval_dataloader(batch_size=256, eval_size=metric_eval_size)
    val_loader = get_retrieval_val_dataloader(batch_size=256, eval_size=metric_eval_size)

    apply_global_l1_pruning(model, amount)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = deepcopy(model.state_dict())
    best_val_recall = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        loss = train_retrieval_epoch(model, train_loader, loss_name, optimizer, device, margin)
        val_metrics = evaluate_retrieval(model, val_loader, device)
        sparsity = masked_weight_sparsity(model)
        print(f"prune amount={amount:.2f} epoch={epoch:02d} sparsity={sparsity:.2f} loss={loss:.4f} val_recall@1={val_metrics['recall@1']:.4f}")
        if val_metrics["recall@1"] >= best_val_recall:
            best_state = deepcopy(model.state_dict())
            best_val_recall = val_metrics["recall@1"]
            best_epoch = epoch

    model.load_state_dict(best_state)
    best_metrics = evaluate_retrieval(model, eval_loader, device)
    remove_pruning_reparametrization(model)
    checkpoint_path = output_dir / f"pruned_global_l1_{amount:.2f}.pt"
    metadata = {
        "stage": "part_c_pruning",
        "method": "global_l1_unstructured",
        "source_checkpoint": teacher_path,
        "backbone": teacher_metadata["backbone"],
        "loss_name": loss_name,
        "embedding_dim": teacher_metadata["embedding_dim"],
        "pruning_amount": amount,
        "best_epoch": best_epoch,
        "retrieval_metrics": best_metrics,
        "dense_parameters": count_parameters(model),
        "nonzero_parameters": count_nonzero_parameters(model),
        "dense_flops": estimate_flops(model),
    }
    save_checkpoint(checkpoint_path, model, metadata)
    row = {
        "method": "global_l1_pruning",
        "amount": amount,
        "metric_augmentation": metric_augmentation,
        "best_epoch": best_epoch,
        "recall@1": best_metrics["recall@1"],
        "recall@5": best_metrics["recall@5"],
        "recall@10": best_metrics["recall@10"],
        "mAP": best_metrics["mAP"],
        "dense_parameters": metadata["dense_parameters"],
        "nonzero_parameters": metadata["nonzero_parameters"],
        "dense_flops": metadata["dense_flops"],
        "checkpoint": str(checkpoint_path),
    }
    append_csv(output_dir / "compression_results.csv", row)
    return row


def distillation_batch_loss(
    student: nn.Module,
    teacher: nn.Module,
    batch: tuple[torch.Tensor, ...],
    loss_name: str,
    device: torch.device,
    margin: float,
    alpha: float,
) -> tuple[torch.Tensor, int]:
    if loss_name == "triplet":
        anchor, positive, negative = [item.to(device) for item in batch]
        images = torch.cat([anchor, positive, negative], dim=0)
        student_embeddings = F.normalize(student(images), p=2, dim=1)
        with torch.no_grad():
            teacher_embeddings = F.normalize(teacher(images), p=2, dim=1)
        student_anchor, student_positive, student_negative = student_embeddings.chunk(3)
        metric_loss = triplet_loss(student_anchor, student_positive, student_negative, margin=margin)
        batch_size = anchor.size(0)
    elif loss_name == "contrastive":
        image_a, image_b, same_class = batch
        image_a = image_a.to(device)
        image_b = image_b.to(device)
        same_class = same_class.to(device)
        images = torch.cat([image_a, image_b], dim=0)
        student_embeddings = F.normalize(student(images), p=2, dim=1)
        with torch.no_grad():
            teacher_embeddings = F.normalize(teacher(images), p=2, dim=1)
        student_a, student_b = student_embeddings.chunk(2)
        metric_loss = contrastive_loss(student_a, student_b, same_class, margin=margin)
        batch_size = image_a.size(0)
    else:
        raise ValueError("Part C expects a triplet or contrastive Part B teacher.")

    geometry_loss = rkd_loss(student_embeddings, teacher_embeddings)
    return (1.0 - alpha) * metric_loss + alpha * geometry_loss, batch_size


def train_distilled_student(
    teacher: nn.Module,
    teacher_path: str,
    teacher_metadata: dict,
    epochs: int,
    learning_rate: float,
    margin: float,
    alpha: float,
    output_dir: Path,
    metric_train_size: int,
    metric_eval_size: int,
    device: torch.device,
    metric_augmentation: bool,
) -> dict[str, float | str | int | bool]:
    set_seed(SEED)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    loss_name = teacher_metadata["loss_name"]
    embedding_dim = teacher_metadata["embedding_dim"]
    student = build_model(COMPRESSION_STUDENT_MODEL, embedding_dim=embedding_dim).to(device)
    train_loader = get_metric_dataloader(
        loss_name=loss_name,
        batch_size=METRIC_BATCH_SIZE,
        subset_size=metric_train_size,
        augment=metric_augmentation,
    )
    eval_loader = get_retrieval_eval_dataloader(batch_size=256, eval_size=metric_eval_size)
    val_loader = get_retrieval_val_dataloader(batch_size=256, eval_size=metric_eval_size)
    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_state = deepcopy(student.state_dict())
    best_val_recall = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, batch_size = distillation_batch_loss(student, teacher, batch, loss_name, device, margin, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        val_metrics = evaluate_retrieval(student, val_loader, device)
        scheduler.step(val_metrics["recall@1"])
        print(f"distill epoch={epoch:02d} loss={total_loss / total_examples:.4f} val_recall@1={val_metrics['recall@1']:.4f} val_recall@5={val_metrics['recall@5']:.4f}")
        if val_metrics["recall@1"] >= best_val_recall:
            best_state = deepcopy(student.state_dict())
            best_val_recall = val_metrics["recall@1"]
            best_epoch = epoch

    student.load_state_dict(best_state)
    best_metrics = evaluate_retrieval(student, eval_loader, device)
    checkpoint_path = output_dir / "distilled_compact_student.pt"
    metadata = {
        "stage": "part_c_distillation",
        "student_model": COMPRESSION_STUDENT_MODEL,
        "teacher_checkpoint": teacher_path,
        "teacher_backbone": teacher_metadata["backbone"],
        "loss_name": loss_name,
        "embedding_dim": embedding_dim,
        "distillation": "rkd_distance_angle",
        "alpha": alpha,
        "metric_augmentation": metric_augmentation,
        "best_epoch": best_epoch,
        "retrieval_metrics": best_metrics,
        "parameters": count_parameters(student),
        "flops": estimate_flops(student),
    }
    save_checkpoint(checkpoint_path, student, metadata)
    row = {
        "method": "distillation_rkd",
        "amount": "",
        "metric_augmentation": metric_augmentation,
        "best_epoch": best_epoch,
        "recall@1": best_metrics["recall@1"],
        "recall@5": best_metrics["recall@5"],
        "recall@10": best_metrics["recall@10"],
        "mAP": best_metrics["mAP"],
        "dense_parameters": metadata["parameters"],
        "nonzero_parameters": count_nonzero_parameters(student),
        "dense_flops": metadata["flops"],
        "checkpoint": str(checkpoint_path),
    }
    append_csv(output_dir / "compression_results.csv", row)
    return row


def select_compressed_model(
    results: list[dict[str, float | str | int | bool]],
    teacher_recall_at_1: float,
    min_preservation: float = 0.90,
) -> dict[str, float | str | int | bool]:
    eligible = [row for row in results if float(row["recall@1"]) >= teacher_recall_at_1 * min_preservation]
    if eligible:
        return min(eligible, key=lambda row: (int(row["dense_flops"]), int(row["dense_parameters"]), -float(row["recall@1"])))
    return max(results, key=lambda row: (float(row["recall@1"]), -int(row["dense_flops"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-checkpoint", default=None)
    parser.add_argument("--pruning-amounts", type=float, nargs="+", default=[0.4])
    parser.add_argument("--prune-epochs", type=int, default=3)
    parser.add_argument("--distill-epochs", type=int, default=8)
    parser.add_argument("--distill-alpha", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--metric-train-size", type=int, default=METRIC_TRAIN_SIZE)
    parser.add_argument("--metric-eval-size", type=int, default=METRIC_EVAL_SIZE)
    parser.add_argument("--output-dir", default="outputs/part_c")
    parser.add_argument("--no-metric-augmentation", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.prune_epochs = min(args.prune_epochs, 1)
        args.distill_epochs = min(args.distill_epochs, 1)
        args.metric_train_size = min(args.metric_train_size, 512)
        args.metric_eval_size = min(args.metric_eval_size, 256)

    output_dir = ensure_dir(args.output_dir)
    results_csv = output_dir / "compression_results.csv"
    if results_csv.exists():
        results_csv.unlink()

    device = get_device()
    teacher_path = resolve_retrieval_checkpoint(args.retrieval_checkpoint)
    teacher, teacher_metadata = load_retrieval_model(teacher_path, device)
    eval_loader = get_retrieval_eval_dataloader(batch_size=256, eval_size=args.metric_eval_size)
    teacher_metrics = evaluate_retrieval(teacher, eval_loader, device)
    print(f"Teacher recall@1={teacher_metrics['recall@1']:.4f} from {teacher_path}")

    results: list[dict[str, float | str | int | bool]] = []
    for amount in args.pruning_amounts:
        results.append(
            train_pruned_model(
                teacher_path=teacher_path,
                teacher_metadata=teacher_metadata,
                amount=amount,
                epochs=args.prune_epochs,
                learning_rate=args.learning_rate,
                margin=args.margin,
                output_dir=output_dir,
                metric_train_size=args.metric_train_size,
                metric_eval_size=args.metric_eval_size,
                device=device,
                metric_augmentation=not args.no_metric_augmentation,
            )
        )

    results.append(
        train_distilled_student(
            teacher=teacher,
            teacher_path=teacher_path,
            teacher_metadata=teacher_metadata,
            epochs=args.distill_epochs,
            learning_rate=args.learning_rate,
            margin=args.margin,
            alpha=args.distill_alpha,
            output_dir=output_dir,
            metric_train_size=args.metric_train_size,
            metric_eval_size=args.metric_eval_size,
            device=device,
            metric_augmentation=not args.no_metric_augmentation,
        )
    )

    write_json(output_dir / "compression_results.json", results)
    selected = select_compressed_model(results, teacher_metrics["recall@1"])
    write_json(output_dir / "selected_compressed_model.json", selected)
    print(f"Selected compressed model: {selected['checkpoint']}")


if __name__ == "__main__":
    main()
