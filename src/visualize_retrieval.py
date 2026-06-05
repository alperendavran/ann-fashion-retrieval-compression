"""
Save qualitative nearest-neighbor retrieval examples for Part B.

Example:
    python src/visualize_retrieval.py --query-indices 0 7 21
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import get_retrieval_eval_dataloader
from models import build_model
from utils import (
    FASHION_CLASSES,
    collect_embeddings,
    denormalize_fashion_images,
    ensure_dir,
    get_device,
    load_checkpoint,
)


def resolve_retrieval_checkpoint(path: str | None) -> str:
    if path:
        return path
    selected_path = Path("outputs/part_b/selected_retrieval_model.json")
    if not selected_path.exists():
        raise FileNotFoundError("Run Part B first or pass --checkpoint.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    return str(selected["checkpoint"])


def load_retrieval_model(path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = load_checkpoint(path, map_location=device)
    metadata = checkpoint["metadata"]
    model = build_model(metadata["backbone"], embedding_dim=metadata["embedding_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def save_retrieval_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    neighbor_indices: torch.Tensor,
    query_index: int,
    output_path: Path,
) -> None:
    selected_indices = [query_index] + neighbor_indices.tolist()
    display_images = denormalize_fashion_images(images).clamp(0.0, 1.0)
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(2.0 * len(selected_indices), 2.4))
    if len(selected_indices) == 1:
        axes = [axes]
    for column, (axis, image_index) in enumerate(zip(axes, selected_indices)):
        axis.imshow(display_images[image_index].squeeze(0), cmap="gray")
        label_name = FASHION_CLASSES[int(labels[image_index])]
        title = "query" if column == 0 else f"nn {column}"
        axis.set_title(f"{title}\n{label_name}", fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/figures/retrieval")
    parser.add_argument("--query-indices", type=int, nargs="+", default=[0, 17, 42])
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--eval-size", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    device = get_device()
    checkpoint_path = resolve_retrieval_checkpoint(args.checkpoint)
    model = load_retrieval_model(checkpoint_path, device)
    eval_loader = get_retrieval_eval_dataloader(batch_size=256, eval_size=args.eval_size)
    embeddings, labels, images = collect_embeddings(model, eval_loader, device, normalize=True)
    distances = torch.cdist(embeddings, embeddings, p=2)
    distances.fill_diagonal_(float("inf"))

    for query_index in args.query_indices:
        if query_index >= len(images):
            raise ValueError(f"Query index {query_index} is outside eval set of size {len(images)}.")
        nearest = distances[query_index].topk(args.neighbors, largest=False).indices
        output_path = output_dir / f"query_{query_index}.png"
        save_retrieval_grid(images, labels, nearest, query_index, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
