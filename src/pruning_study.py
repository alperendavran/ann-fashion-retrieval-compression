"""
Pruning-method comparison on the Part B teacher.

Separate from the main Part C selection. It mirrors
the course pruning exercise on the Part B retrieval teacher:

1. Compare the four taught pruning methods (local/global x L1/random) at a single
   sparsity, with retrieval fine-tuning, to confirm the expected ordering
   (L1 > random, global > local).
2. Sweep global L1 sparsity to study how much of the retrieval model is
   over-parameterized (how far it can be pruned before Recall@1 degrades).

The main pipeline keeps global L1 at 40% as the reported pruning row; this study
only adds supporting evidence and does not change the selected compressed model.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.utils.prune as prune

from config import METRIC_EVAL_SIZE, METRIC_TRAIN_SIZE, SEED
from data import get_metric_dataloader, get_retrieval_eval_dataloader, get_retrieval_val_dataloader
from prune_and_distill import (
    evaluate_retrieval,
    load_retrieval_model,
    prunable_modules,
    remove_pruning_reparametrization,
    resolve_retrieval_checkpoint,
    train_retrieval_epoch,
)
from utils import ensure_dir, get_device, set_seed, write_json


def local_l1_prune(model: torch.nn.Module, amount: float) -> None:
    for module, name in prunable_modules(model):
        prune.l1_unstructured(module, name, amount)


def local_random_prune(model: torch.nn.Module, amount: float) -> None:
    for module, name in prunable_modules(model):
        prune.random_unstructured(module, name, amount)


def global_l1_prune(model: torch.nn.Module, amount: float) -> None:
    prune.global_unstructured(prunable_modules(model), pruning_method=prune.L1Unstructured, amount=amount)


def global_random_prune(model: torch.nn.Module, amount: float) -> None:
    prune.global_unstructured(prunable_modules(model), pruning_method=prune.RandomUnstructured, amount=amount)


PRUNE_METHODS = {
    "local_l1": local_l1_prune,
    "local_random": local_random_prune,
    "global_l1": global_l1_prune,
    "global_random": global_random_prune,
}


def realized_sparsity(model: torch.nn.Module) -> float:
    total = 0
    nonzero = 0
    for module, name in prunable_modules(model):
        weight = getattr(module, name)
        total += weight.numel()
        nonzero += int(torch.count_nonzero(weight).item())
    return 1.0 - nonzero / total if total else 0.0


def run_one(
    method: str,
    amount: float,
    teacher_path: str,
    loss_name: str,
    epochs: int,
    learning_rate: float,
    margin: float,
    train_size: int,
    augment: bool,
    eval_loader,
    val_loader,
    device: torch.device,
) -> dict[str, float | str]:
    # Re-seed and rebuild the loader for every run so each fine-tune is fully
    # deterministic and identical settings (e.g. global L1 @ 40%) reproduce exactly.
    set_seed(SEED)
    train_loader = get_metric_dataloader(
        loss_name=loss_name, batch_size=64, subset_size=train_size, augment=augment
    )
    model, _ = load_retrieval_model(teacher_path, device)
    PRUNE_METHODS[method](model, amount)
    sparsity = realized_sparsity(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state = deepcopy(model.state_dict())
    best_val_recall = -1.0
    for _ in range(1, epochs + 1):
        train_retrieval_epoch(model, train_loader, loss_name, optimizer, device, margin)
        val_recall = evaluate_retrieval(model, val_loader, device)["recall@1"]
        if val_recall >= best_val_recall:
            best_val_recall = val_recall
            best_state = deepcopy(model.state_dict())

    # Report the disjoint test set once on the validation-selected checkpoint.
    model.load_state_dict(best_state)
    test_recall = evaluate_retrieval(model, eval_loader, device)["recall@1"]
    remove_pruning_reparametrization(model)
    return {
        "method": method,
        "requested_amount": amount,
        "realized_sparsity": round(sparsity, 4),
        "recall@1": round(test_recall, 4),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-checkpoint", default=None)
    parser.add_argument("--compare-amount", type=float, default=0.4)
    parser.add_argument("--sweep-amounts", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--metric-train-size", type=int, default=METRIC_TRAIN_SIZE)
    parser.add_argument("--metric-eval-size", type=int, default=METRIC_EVAL_SIZE)
    parser.add_argument("--output-dir", default="outputs/part_c")
    parser.add_argument("--no-metric-augmentation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    teacher_path = resolve_retrieval_checkpoint(args.retrieval_checkpoint)
    _, metadata = load_retrieval_model(teacher_path, device)
    loss_name = metadata["loss_name"]

    eval_loader = get_retrieval_eval_dataloader(batch_size=256, eval_size=args.metric_eval_size)
    val_loader = get_retrieval_val_dataloader(batch_size=256, eval_size=args.metric_eval_size)
    teacher_recall = evaluate_retrieval(load_retrieval_model(teacher_path, device)[0], eval_loader, device)["recall@1"]
    print(f"Teacher recall@1={teacher_recall:.4f}")

    common = dict(
        teacher_path=teacher_path,
        loss_name=loss_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        margin=args.margin,
        train_size=args.metric_train_size,
        augment=not args.no_metric_augmentation,
        eval_loader=eval_loader,
        val_loader=val_loader,
        device=device,
    )

    comparison = []
    for method in PRUNE_METHODS:
        row = run_one(method, args.compare_amount, **common)
        print(f"[compare@{args.compare_amount:.0%}] {method:14s} sparsity={row['realized_sparsity']:.3f} recall@1={row['recall@1']:.4f}")
        comparison.append(row)

    sweep = []
    for amount in args.sweep_amounts:
        row = run_one("global_l1", amount, **common)
        print(f"[sweep global_l1] amount={amount:.0%} sparsity={row['realized_sparsity']:.3f} recall@1={row['recall@1']:.4f}")
        sweep.append(row)

    output = {
        "teacher_recall@1": round(teacher_recall, 4),
        "compare_amount": args.compare_amount,
        "method_comparison": comparison,
        "global_l1_sweep": sweep,
        "fine_tune_epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }
    out_path = ensure_dir(args.output_dir) / "pruning_study.json"
    write_json(out_path, output)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
