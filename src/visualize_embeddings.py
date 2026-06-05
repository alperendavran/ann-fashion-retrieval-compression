"""
Save t-SNE and diagnostic visualizations of retrieval embeddings.

Example:
    python src/visualize_embeddings.py --max-points 1000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from data import get_retrieval_eval_dataloader
from models import build_model
from utils import (
    FASHION_CLASSES,
    collect_embeddings,
    embedding_geometry_diagnostics,
    ensure_dir,
    get_device,
    load_checkpoint,
    per_class_recall_at_k,
    retrieval_confusion_matrix,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-path", default="outputs/figures/embedding_tsne.png")
    parser.add_argument("--confusion-output-path", default="outputs/figures/retrieval_confusion_matrix.png")
    parser.add_argument("--diagnostics-path", default="outputs/part_b/retrieval_diagnostics.json")
    parser.add_argument("--per-class-path", default="outputs/part_b/per_class_recall_at_1.csv")
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--perplexity", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    checkpoint_path = resolve_retrieval_checkpoint(args.checkpoint)
    model = load_retrieval_model(checkpoint_path, device)
    eval_loader = get_retrieval_eval_dataloader(batch_size=256, eval_size=args.eval_size)
    full_embeddings, full_labels, _images = collect_embeddings(model, eval_loader, device, normalize=True)

    tsne_embeddings = full_embeddings[: args.max_points].numpy()
    tsne_labels = full_labels[: args.max_points].numpy()
    perplexity = min(args.perplexity, max(5.0, (len(tsne_embeddings) - 1) / 3))
    reduced = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(tsne_embeddings)

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    fig, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(reduced[:, 0], reduced[:, 1], c=tsne_labels, cmap="tab10", s=12, alpha=0.8)
    handles, _ = scatter.legend_elements(num=10)
    axis.legend(handles, FASHION_CLASSES, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    axis.set_title("Retrieval Embedding Space (t-SNE)")
    axis.set_xticks([])
    axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")

    full_per_class_recall = per_class_recall_at_k(full_embeddings, full_labels, k=1)
    confusion = retrieval_confusion_matrix(full_embeddings, full_labels)
    geometry = embedding_geometry_diagnostics(full_embeddings, full_labels)

    diagnostics_path = Path(args.diagnostics_path)
    ensure_dir(diagnostics_path.parent)
    diagnostics_path.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint_path,
                "eval_size": int(full_labels.numel()),
                "tsne_max_points": int(min(args.max_points, full_labels.numel())),
                "tsne_perplexity": float(perplexity),
                "tsne_random_state": 42,
                "recall_at_1_definition": (
                    "nearest neighbor by L2 distance in L2-normalized embedding space, "
                    "self excluded, hit if same FashionMNIST class"
                ),
                "per_class_recall_at_1": full_per_class_recall,
                "geometry_diagnostics": geometry,
                "confusion_matrix_rows_true_cols_retrieved": confusion.tolist(),
                "classes": list(FASHION_CLASSES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {diagnostics_path}")

    per_class_path = Path(args.per_class_path)
    ensure_dir(per_class_path.parent)
    with per_class_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "recall@1"])
        writer.writeheader()
        for class_name, value in full_per_class_recall.items():
            writer.writerow({"class_name": class_name, "recall@1": value})
    print(f"Saved {per_class_path}")

    confusion_output_path = Path(args.confusion_output_path)
    ensure_dir(confusion_output_path.parent)
    row_totals = confusion.sum(dim=1, keepdim=True).clamp(min=1)
    normalized_confusion = confusion.float() / row_totals
    fig, axis = plt.subplots(figsize=(8, 6.8))
    image = axis.imshow(normalized_confusion.numpy(), cmap="Reds", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(FASHION_CLASSES)))
    axis.set_yticks(range(len(FASHION_CLASSES)))
    axis.set_xticklabels(FASHION_CLASSES, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(FASHION_CLASSES, fontsize=8)
    axis.set_xlabel("Retrieved nearest-neighbor class")
    axis.set_ylabel("Query class")
    axis.set_title("Top-1 Retrieval Confusion Matrix")
    for row in range(len(FASHION_CLASSES)):
        for column in range(len(FASHION_CLASSES)):
            value = normalized_confusion[row, column].item()
            if value >= 0.08 or row == column:
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > 0.55 else "black",
                )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(confusion_output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {confusion_output_path}")


if __name__ == "__main__":
    main()
