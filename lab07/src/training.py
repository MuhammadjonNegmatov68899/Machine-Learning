"""
Training pipelines for baseline CNN, fine-tuning of pruned models, and student training.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lab07.src.pruning import (
    apply_pruning_masks,
    zero_masked_gradients,
    verify_mask_adherence,
)
from lab07.src.distillation import DistillationLoss, train_distillation_epoch


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates model on dataloader.
    Returns:
        loss: Average cross-entropy loss.
        accuracy: Accuracy as fraction in [0.0, 1.0].
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[float, float]:
    """
    Executes one standard training epoch.
    If masks are provided, gradients and weights for masked entries are kept strictly zero.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for data, targets in dataloader:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        if masks is not None:
            zero_masked_gradients(model, masks)

        optimizer.step()

        if masks is not None:
            apply_pruning_masks(model, masks)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Standard training routine for baseline or student models.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    start_total_time = time.perf_counter()
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                    },
                    save_path,
                )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "epoch_time_sec": round(epoch_time, 2),
            }
        )

    total_training_time = time.perf_counter() - start_total_time

    # Evaluate final model on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    return {
        "history": history,
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 4),
        "total_training_time_sec": round(total_training_time, 2),
        "best_val_acc": round(best_val_acc, 4),
    }


def finetune_pruned_model(
    model: nn.Module,
    masks: Optional[Dict[str, torch.Tensor]],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 3,
    lr: float = 0.0005,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Fine-tunes a pruned model for exactly 3 epochs.
    Guarantees that pruning masks are strictly adhered to throughout training.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Pre-finetune evaluation
    pre_loss, pre_acc = evaluate(model, test_loader, criterion, device)

    if masks is not None:
        assert verify_mask_adherence(model, masks), "Mask violated prior to fine-tuning!"

    history: List[Dict[str, Any]] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, masks=masks
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.perf_counter() - t0

        if masks is not None:
            assert verify_mask_adherence(
                model, masks
            ), f"Mask adherence failed at epoch {epoch}!"

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "epoch_time_sec": round(epoch_time, 2),
            }
        )

    total_time = time.perf_counter() - start_time
    post_loss, post_acc = evaluate(model, test_loader, criterion, device)

    if masks is not None:
        assert verify_mask_adherence(
            model, masks
        ), "Final mask adherence verification failed!"

    return {
        "pre_finetune_test_acc": round(pre_acc, 4),
        "post_finetune_test_acc": round(post_acc, 4),
        "pre_finetune_test_loss": round(pre_loss, 4),
        "post_finetune_test_loss": round(post_loss, 4),
        "finetune_time_sec": round(total_time, 2),
        "history": history,
    }


def train_distilled_student(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trains a student model using Knowledge Distillation from a frozen teacher model.
    """
    if device is None:
        device = next(student.parameters()).device

    student.to(device)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    distill_loss_fn = DistillationLoss(temperature=temperature, alpha=alpha)
    eval_criterion = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    start_total_time = time.perf_counter()
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc, detailed = train_distillation_epoch(
            student, teacher, train_loader, optimizer, distill_loss_fn, device
        )
        val_loss, val_acc = evaluate(student, val_loader, eval_criterion, device)
        epoch_time = time.perf_counter() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "student_state_dict": student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "temperature": temperature,
                        "alpha": alpha,
                    },
                    save_path,
                )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "ce_loss": round(detailed["ce_loss"], 4),
                "kl_loss": round(detailed["kl_loss"], 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "epoch_time_sec": round(epoch_time, 2),
            }
        )

    total_training_time = time.perf_counter() - start_total_time
    test_loss, test_acc = evaluate(student, test_loader, eval_criterion, device)

    return {
        "history": history,
        "test_loss": round(test_loss, 4),
        "test_acc": round(test_acc, 4),
        "total_training_time_sec": round(total_training_time, 2),
        "best_val_acc": round(best_val_acc, 4),
    }
