"""
Knowledge Distillation module for Lab 07:
Implements teacher-student transfer using soft target distribution matching.
"""

from typing import Dict, Any, Tuple, Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining hard label Cross-Entropy with soft target KL-Divergence.
    
    Formula:
        Loss = alpha * CE(student_logits, labels) +
               (1 - alpha) * T^2 * KL(softmax(student_logits / T) || softmax(teacher_logits / T))
               
    Numerically stable implementation using log_softmax and target probabilities.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        # Using reduction='batchmean' as mathematically defined for KL divergence over batches
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the blended distillation loss.
        
        Args:
            student_logits: Unnormalized predictions from the student model (B, num_classes).
            teacher_logits: Unnormalized predictions from the frozen teacher model (B, num_classes).
            labels: Ground truth categorical labels (B,).
            
        Returns:
            total_loss: Scalar loss tensor with attached gradient.
            metrics: Dictionary of individual loss components for logging.
        """
        t = self.temperature

        # Hard target loss
        ce_loss = self.ce_loss_fn(student_logits, labels)

        # Soft target loss with temperature scaling
        # PyTorch KLDivLoss expects input as log-probabilities and target as probabilities
        student_log_probs = F.log_softmax(student_logits / t, dim=-1)
        teacher_probs = F.softmax(teacher_logits / t, dim=-1)

        # Scale by T^2 to balance gradient magnitudes as derived by Hinton et al.
        kl_loss = self.kl_loss_fn(student_log_probs, teacher_probs) * (t * t)

        total_loss = self.alpha * ce_loss + (1.0 - self.alpha) * kl_loss

        metrics = {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "kl_loss": kl_loss.item(),
        }

        return total_loss, metrics


def train_distillation_epoch(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    distill_loss_fn: DistillationLoss,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Executes one training epoch with knowledge distillation.
    Teacher is strictly frozen and evaluated in no_grad mode.
    """
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    correct = 0
    total_samples = 0

    for data, targets in train_loader:
        data = data.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            teacher_logits = teacher(data)

        optimizer.zero_grad()
        student_logits = student(data)

        loss, metrics = distill_loss_fn(student_logits, teacher_logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_ce += metrics["ce_loss"] * batch_size
        total_kl += metrics["kl_loss"] * batch_size

        preds = student_logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    detailed_metrics = {
        "loss": avg_loss,
        "ce_loss": total_ce / total_samples,
        "kl_loss": total_kl / total_samples,
        "accuracy": avg_acc,
    }

    return avg_loss, avg_acc, detailed_metrics
