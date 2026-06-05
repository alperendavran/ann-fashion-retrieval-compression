"""
Shared FashionMNIST data protocol for the project.

The split seed and subset sizes are fixed in config.py so Part A, Part B, and
Part C stay comparable. Training scripts may change batch size, but should not
silently change the subset protocol.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from config import (
    CLASSIFICATION_TRAIN_SIZE,
    CLASSIFICATION_VAL_SIZE,
    DATA_ROOT,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    METRIC_EVAL_SIZE,
    METRIC_TRAIN_SIZE,
    METRIC_VAL_SIZE,
    SEED,
)


def _fixed_indices(dataset_size: int, take: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(dataset_size, generator=generator).tolist()[:take]


def _loader_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _indices_by_class(targets: Sequence[int]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(targets):
        result[int(label)].append(index)
    return result


def fashion_mnist_transform(train: bool, augment: bool = True) -> transforms.Compose:
    steps: list[object] = []
    if train and augment:
        steps.extend([transforms.RandomHorizontalFlip(), transforms.RandomCrop(28, padding=2)])
    steps.extend([transforms.ToTensor(), transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD)])
    return transforms.Compose(steps)


def get_classification_dataloaders(
    batch_size: int = 64,
    train_size: int = CLASSIFICATION_TRAIN_SIZE,
    val_size: int = CLASSIFICATION_VAL_SIZE,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=fashion_mnist_transform(train=True),
    )
    val_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=fashion_mnist_transform(train=False),
    )
    test_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=fashion_mnist_transform(train=False),
    )

    indices = _fixed_indices(len(train_dataset), train_size + val_size, seed)
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=_loader_generator(seed),
    )
    val_loader = DataLoader(Subset(val_dataset, val_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class ContrastiveFashionDataset(Dataset):
    """Deterministic positive/negative pairs for contrastive retrieval training."""

    def __init__(
        self,
        subset_size: int = METRIC_TRAIN_SIZE,
        seed: int = SEED,
        augment: bool = True,
    ) -> None:
        base = datasets.FashionMNIST(
            root=DATA_ROOT,
            train=True,
            download=True,
            transform=fashion_mnist_transform(train=True, augment=augment),
        )
        indices = _fixed_indices(len(base), subset_size, seed)
        self.base = Subset(base, indices)
        self.targets = [int(base.targets[index]) for index in indices]
        self.by_class = _indices_by_class(self.targets)

        rng = random.Random(seed)
        self.pairs: list[tuple[int, int]] = []
        for index, label in enumerate(self.targets):
            if rng.random() < 0.5:
                candidates = [item for item in self.by_class[label] if item != index]
                pair_index = rng.choice(candidates or self.by_class[label])
            else:
                negative_label = rng.choice([class_id for class_id in self.by_class if class_id != label])
                pair_index = rng.choice(self.by_class[negative_label])
            self.pairs.append((index, pair_index))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        first_index, second_index = self.pairs[index]
        first_image, first_label = self.base[first_index]
        second_image, second_label = self.base[second_index]
        same_class = float(first_label == second_label)
        return first_image, second_image, torch.tensor(same_class, dtype=torch.float32)


class TripletFashionDataset(Dataset):
    """Deterministic anchor/positive/negative triplets for Part B and Part C."""

    def __init__(
        self,
        subset_size: int = METRIC_TRAIN_SIZE,
        seed: int = SEED,
        augment: bool = True,
    ) -> None:
        base = datasets.FashionMNIST(
            root=DATA_ROOT,
            train=True,
            download=True,
            transform=fashion_mnist_transform(train=True, augment=augment),
        )
        indices = _fixed_indices(len(base), subset_size, seed)
        self.base = Subset(base, indices)
        self.targets = [int(base.targets[index]) for index in indices]
        self.by_class = _indices_by_class(self.targets)

        rng = random.Random(seed + 1)
        self.triplets: list[tuple[int, int, int]] = []
        for index, label in enumerate(self.targets):
            positives = [item for item in self.by_class[label] if item != index]
            positive_index = rng.choice(positives or self.by_class[label])
            negative_label = rng.choice([class_id for class_id in self.by_class if class_id != label])
            negative_index = rng.choice(self.by_class[negative_label])
            self.triplets.append((index, positive_index, negative_index))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        anchor_index, positive_index, negative_index = self.triplets[index]
        anchor, _ = self.base[anchor_index]
        positive, _ = self.base[positive_index]
        negative, _ = self.base[negative_index]
        return anchor, positive, negative


def get_metric_dataloader(
    loss_name: str,
    batch_size: int = 64,
    subset_size: int = METRIC_TRAIN_SIZE,
    seed: int = SEED,
    augment: bool = True,
) -> DataLoader:
    if loss_name == "triplet":
        dataset = TripletFashionDataset(subset_size=subset_size, seed=seed, augment=augment)
    elif loss_name == "contrastive":
        dataset = ContrastiveFashionDataset(subset_size=subset_size, seed=seed, augment=augment)
    else:
        raise ValueError("Part B supports only 'triplet' or 'contrastive' for the brief.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=_loader_generator(seed),
    )


def get_metric_label_dataloader(
    batch_size: int = 128,
    subset_size: int = METRIC_TRAIN_SIZE,
    seed: int = SEED,
    augment: bool = True,
) -> DataLoader:
    """
    (image, label) batches over the same fixed metric-training subset.

    Used for online batch-hard triplet mining, where the hardest positive and
    negative are selected inside each batch instead of using pre-sampled random
    triplets. The subset indices match the triplet/contrastive datasets so the
    training protocol stays comparable.
    """
    base = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=fashion_mnist_transform(train=True, augment=augment),
    )
    indices = _fixed_indices(len(base), subset_size, seed)
    return DataLoader(
        Subset(base, indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=_loader_generator(seed),
    )


def get_retrieval_eval_dataloader(
    batch_size: int = 256,
    eval_size: int = METRIC_EVAL_SIZE,
    seed: int = SEED,
) -> DataLoader:
    test_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=fashion_mnist_transform(train=False),
    )
    eval_indices = _fixed_indices(len(test_dataset), eval_size, seed)
    return DataLoader(Subset(test_dataset, eval_indices), batch_size=batch_size, shuffle=False)


def get_retrieval_val_dataloader(
    batch_size: int = 256,
    val_size: int = METRIC_VAL_SIZE,
    eval_size: int = METRIC_EVAL_SIZE,
    seed: int = SEED,
) -> DataLoader:
    """
    Validation split for checkpoint/epoch selection.

    It uses the same fixed permutation as the reported eval loader but takes the
    slice *after* the eval indices, so the validation set is disjoint from the
    reported retrieval test set. Model selection therefore never touches the
    numbers that are reported.
    """
    test_dataset = datasets.FashionMNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=fashion_mnist_transform(train=False),
    )
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(test_dataset), generator=generator).tolist()
    val_indices = permutation[eval_size:eval_size + val_size]
    return DataLoader(Subset(test_dataset, val_indices), batch_size=batch_size, shuffle=False)
