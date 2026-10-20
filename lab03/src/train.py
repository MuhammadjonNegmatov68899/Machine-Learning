"""
Training module for Fashion-MNIST classification across MLP, CNN, and Attention models.
"""

from typing import Dict, Any, List, Tuple
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates loss and accuracy on a given DataLoader.
    
    Returns:
        (average_loss, accuracy_percentage)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * (correct / total_samples)
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains a model for exactly `epochs` using Adam with `lr`.
    
    Logs:
        - Train loss
        - Val loss
        - Val accuracy
        - Final test loss and accuracy
    """
    set_seed(seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_sec": [],
    }

    print(f"\n--- Training {model.__class__.__name__} on {device} ({epochs} epochs, lr={lr}) ---")

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

        epoch_time = time.time() - t0
        epoch_train_loss = running_train_loss / train_samples

        # Validation evaluation (strictly for monitoring, never for test selection)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_sec"].append(epoch_time)

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:6.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    # Final evaluation on the official test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Evaluation: Loss = {test_loss:.4f}, Accuracy = {test_acc:.2f}%\n")

    return {
        "model": model,
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
