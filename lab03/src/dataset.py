"""
Dataset module for Fashion-MNIST loading, normalization, and splitting.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_fashion_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads Fashion-MNIST, normalizes it using the dataset mean (0.2860) and std (0.3530),
    splits the official 60k training set into 55k training and 5k validation sets,
    and returns DataLoaders for train, val, and the official 10k test set.

    Args:
        data_dir: Local path where the dataset should be stored/downloaded.
        batch_size: Mini-batch size (default: 128).
        seed: Random seed for reproducible splitting.
        num_workers: Number of DataLoader worker subprocesses.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Transform: ToTensor() followed by Normalize((0.2860,), (0.3530,))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    # Official training set (60,000 images)
    full_train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Official test set (10,000 images)
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Deterministic split: 55,000 train, 5,000 validation
    train_subset, val_subset = random_split(
        full_train_dataset,
        [55000, 5000],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
