"""
Grad-CAM on the selected Part A classifier.

Grad-CAM (Selvaraju et al., 2017) builds a class-specific heatmap from the
last convolutional feature map and the gradients of the predicted-class score.
This script only visualizes the model; it does not retrain anything.

Used to inspect why upper-garment classes (T-shirt, Pullover, Coat, Shirt)
are harder for retrieval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from config import DATA_ROOT, FASHION_MNIST_MEAN, FASHION_MNIST_STD
from models import build_model
from utils import ensure_dir, get_device, load_checkpoint, set_seed

CLASS_NAMES = [
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
]

# Default panel: the four confusable upper garments plus four easy classes.
DEFAULT_CLASSES = [0, 2, 4, 6, 1, 7, 8, 9]


def resolve_classifier_checkpoint(path: str | None) -> str:
    if path:
        return path
    selected = Path("outputs/part_a/selected_backbone.json")
    if not selected.exists():
        raise FileNotFoundError("Run Part A first or pass --classifier-checkpoint.")
    return str(json.loads(selected.read_text(encoding="utf-8"))["checkpoint"])


def load_classifier(path: str, device: torch.device) -> tuple[torch.nn.Module, str]:
    checkpoint = load_checkpoint(path, map_location=device)
    metadata = checkpoint["metadata"]
    name = metadata["model_name"]
    model = build_model(name, num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, name


def grad_cam(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Return a [0,1] heatmap upsampled to the input size and the predicted class."""
    # The last entry in `.features` is the global pool; the one before it is the
    # last convolutional feature map for every backbone in this project.
    target_layer = model.features[-2]

    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        captured["activations"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradients", grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        logits = model(image.to(device))
        predicted = int(logits.argmax(dim=1).item())
        model.zero_grad(set_to_none=True)
        logits[0, predicted].backward()

        activations = captured["activations"]  # (1, C, h, w)
        gradients = captured["gradients"]  # (1, C, h, w)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # channel importance
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    finally:
        handle.remove()
    return cam, predicted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--output", default="outputs/figures/gradcam.png")
    args = parser.parse_args()

    set_seed(0)
    device = get_device()
    checkpoint_path = resolve_classifier_checkpoint(args.classifier_checkpoint)
    model, name = load_classifier(checkpoint_path, device)
    print(f"Grad-CAM on {name} from {checkpoint_path}")

    raw_dataset = datasets.FashionMNIST(
        root=DATA_ROOT, train=False, download=True, transform=transforms.ToTensor()
    )
    normalize = transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD)

    first_index_for: dict[int, int] = {}
    for index in range(len(raw_dataset)):
        label = int(raw_dataset.targets[index])
        if label in args.classes and label not in first_index_for:
            first_index_for[label] = index
        if len(first_index_for) == len(args.classes):
            break

    columns = 4
    rows = (len(args.classes) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 2.8 * rows))
    axes = axes.flatten()

    for axis, target_class in zip(axes, args.classes):
        index = first_index_for[target_class]
        raw_image, true_label = raw_dataset[index]
        model_input = normalize(raw_image).unsqueeze(0)
        cam, predicted = grad_cam(model, model_input, device)

        axis.imshow(raw_image.squeeze(), cmap="gray")
        axis.imshow(cam, cmap="jet", alpha=0.45)
        correct = predicted == true_label
        color = "green" if correct else "red"
        axis.set_title(
            f"true: {CLASS_NAMES[true_label]}\npred: {CLASS_NAMES[predicted]}",
            fontsize=9,
            color=color,
        )
        axis.axis("off")

    for axis in axes[len(args.classes):]:
        axis.axis("off")

    fig.suptitle(f"Grad-CAM heatmaps ({name})", fontsize=12)
    fig.tight_layout()
    output_path = ensure_dir(Path(args.output).parent) / Path(args.output).name
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
