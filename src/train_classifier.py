"""
Part A: train and compare the standard CNN and depthwise separable CNN.

Example:
    python src/train_classifier.py --epochs 8 --learning-rates 0.001 0.0005
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from config import CLASSIFICATION_BATCH_SIZE, PART_A_MODELS, SEED
from data import get_classification_dataloaders
from models import build_model
from utils import (
    append_csv,
    classification_metrics,
    count_parameters,
    ensure_dir,
    estimate_flops,
    get_device,
    save_checkpoint,
    set_seed,
    write_json,
)


def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    all_logits = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits_cpu = torch.cat(all_logits)
    targets_cpu = torch.cat(all_targets)
    metrics = classification_metrics(logits_cpu, targets_cpu)
    metrics["loss"] = total_loss / total_examples
    return metrics


def train_one_configuration(
    model_name: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    output_dir: Path,
    train_size: int | None,
    val_size: int | None,
    device: torch.device,
    early_stop_patience: int = 0,
) -> dict[str, float | str | int]:
    set_seed(SEED)
    loader_kwargs = {"batch_size": batch_size}
    if train_size is not None:
        loader_kwargs["train_size"] = train_size
    if val_size is not None:
        loader_kwargs["val_size"] = val_size
    train_loader, val_loader, test_loader = get_classification_dataloaders(**loader_kwargs)

    model = build_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    best_state = deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["macro_f1"])
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{model_name} lr={learning_rate:g} epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"current_lr={current_lr:.2e}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stop_patience and epochs_without_improvement >= early_stop_patience:
                print(f"early_stop epoch={epoch:02d} patience={early_stop_patience}")
                break

    model.load_state_dict(best_state)
    val_metrics = run_epoch(model, val_loader, criterion, device)
    test_metrics = run_epoch(model, test_loader, criterion, device)
    params = count_parameters(model)
    flops = estimate_flops(model)

    checkpoint_path = output_dir / f"{model_name}_lr{learning_rate:g}_best.pt"
    metadata = {
        "stage": "part_a_classifier",
        "model_name": model_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "parameters": params,
        "flops": flops,
    }
    save_checkpoint(checkpoint_path, model, metadata)

    row = {
        "model": model_name,
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "parameters": params,
        "flops": flops,
        "checkpoint": str(checkpoint_path),
    }
    append_csv(output_dir / "classifier_results.csv", row)
    return row


def select_backbone(results: list[dict[str, float | str | int]]) -> dict[str, float | str | int]:
    return max(results, key=lambda row: (float(row["val_macro_f1"]), -int(row["flops"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=CLASSIFICATION_BATCH_SIZE)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-3, 5e-4])
    parser.add_argument("--models", nargs="+", default=list(PART_A_MODELS), choices=list(PART_A_MODELS))
    parser.add_argument("--output-dir", default="outputs/part_a")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Use tiny subsets for a fast smoke run.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after N epochs without validation macro-F1 improvement. 0 disables early stopping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.train_size = args.train_size or 512
        args.val_size = args.val_size or 256

    output_dir = ensure_dir(args.output_dir)
    results_csv = output_dir / "classifier_results.csv"
    if results_csv.exists():
        results_csv.unlink()
    device = get_device()
    print(f"Using device: {device}")

    results = []
    for model_name in args.models:
        for learning_rate in args.learning_rates:
            row = train_one_configuration(
                model_name=model_name,
                learning_rate=learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                output_dir=output_dir,
                train_size=args.train_size,
                val_size=args.val_size,
                device=device,
                early_stop_patience=args.early_stop_patience,
            )
            results.append(row)

    selected = select_backbone(results)
    write_json(output_dir / "classifier_results.json", results)
    write_json(output_dir / "selected_backbone.json", selected)
    print(f"Selected Part A backbone: {selected['model']} from {selected['checkpoint']}")


if __name__ == "__main__":
    main()
