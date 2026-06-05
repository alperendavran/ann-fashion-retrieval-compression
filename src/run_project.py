"""
Run the required project stages in order: Part A -> Part B -> Part C.

Quick smoke run:
    python src/run_project.py --quick

Full run:
    python src/run_project.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run_command(args: list[str]) -> None:
    print("\n$ " + " ".join(args))
    subprocess.check_call(args, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Tiny subsets and one epoch for smoke testing.")
    parser.add_argument("--classifier-epochs", type=int, default=8)
    parser.add_argument("--metric-epochs", type=int, default=8)
    parser.add_argument("--prune-epochs", type=int, default=5)
    parser.add_argument("--distill-epochs", type=int, default=10)
    parser.add_argument("--skip-visuals", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    quick = ["--quick"] if args.quick else []

    run_command(
        [
            python,
            str(SRC / "train_classifier.py"),
            "--epochs",
            str(args.classifier_epochs),
            "--learning-rates",
            "0.001",
            "0.0005",
            *quick,
        ]
    )
    run_command(
        [
            python,
            str(SRC / "train_metric.py"),
            "--epochs",
            str(args.metric_epochs),
            "--losses",
            "triplet",
            "--learning-rate",
            "5e-5",
            "--batch-size",
            "256",
            "--no-metric-augmentation",
            *quick,
        ]
    )
    run_command(
        [
            python,
            str(SRC / "prune_and_distill.py"),
            "--prune-epochs",
            str(args.prune_epochs),
            "--distill-epochs",
            str(args.distill_epochs),
            "--pruning-amounts",
            "0.4",
            "--distill-alpha",
            "0.9",
            "--no-metric-augmentation",
            *quick,
        ]
    )

    if not args.skip_visuals:
        visual_eval_size = "256" if args.quick else "2000"
        run_command([python, str(SRC / "visualize_retrieval.py"), "--eval-size", visual_eval_size])
        run_command([python, str(SRC / "visualize_embeddings.py"), "--eval-size", visual_eval_size])

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
