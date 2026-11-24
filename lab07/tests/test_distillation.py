"""
Tests for Knowledge Distillation loss, numerical stability, and teacher freezing.
"""

import pytest
import torch
import torch.nn as nn
from lab07.src.distillation import DistillationLoss, train_distillation_epoch
from lab07.src.models import BaselineCNN, StudentCNN


def test_distillation_loss_values():
    """Verifies that DistillationLoss computes correct scalar loss and sub-metrics."""
    distill_fn = DistillationLoss(temperature=4.0, alpha=0.5)

    student_logits = torch.tensor([[2.0, -1.0, 0.5]], requires_grad=True)
    teacher_logits = torch.tensor([[1.8, -0.9, 0.4]])
    labels = torch.tensor([0])

    loss, metrics = distill_fn(student_logits, teacher_logits, labels)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0
    assert "ce_loss" in metrics
    assert "kl_loss" in metrics

    # Test backward pass
    loss.backward()
    assert student_logits.grad is not None
    assert not torch.isnan(student_logits.grad).any()


def test_teacher_freezing_behavior(baseline_model, student_model, dummy_input):
    """
    Verifies that the teacher model is strictly frozen and accumulates no gradients
    during student distillation backward steps.
    """
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad = False

    distill_fn = DistillationLoss(temperature=4.0, alpha=0.5)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01)

    labels = torch.randint(0, 10, (dummy_input.size(0),))

    with torch.no_grad():
        teacher_logits = baseline_model(dummy_input)

    optimizer.zero_grad()
    student_logits = student_model(dummy_input)
    loss, _ = distill_fn(student_logits, teacher_logits, labels)
    loss.backward()

    # Assert student parameters have gradients
    has_student_grads = any(
        p.grad is not None and p.grad.norm().item() > 0 for p in student_model.parameters()
    )
    assert has_student_grads, "Student failed to accumulate gradients!"

    # Assert teacher parameters strictly have NO gradients
    for name, p in baseline_model.named_parameters():
        assert p.grad is None, f"Teacher parameter {name} erroneously received gradients!"


def test_numerical_stability_extreme_logits():
    """Verifies numerical stability when logits contain large or small numbers."""
    distill_fn = DistillationLoss(temperature=4.0, alpha=0.5)

    large_student = torch.full((4, 10), 100.0, requires_grad=True)
    large_teacher = torch.full((4, 10), -100.0)
    labels = torch.zeros(4, dtype=torch.long)

    loss, _ = distill_fn(large_student, large_teacher, labels)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_student_evaluation_and_checkpoint_loading(tmp_path):
    """Verifies that evaluate function works properly with StudentCNN and checkpoint loading."""
    from lab07.src.training import evaluate
    from torch.utils.data import TensorDataset, DataLoader

    student = StudentCNN()
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    crit = nn.CrossEntropyLoss()

    loss, acc = evaluate(student, loader, crit, torch.device("cpu"))
    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

    # Save and reload checkpoint
    ckpt_path = tmp_path / "test_student.pt"
    torch.save({"model_state_dict": student.state_dict(), "val_acc": acc}, ckpt_path)

    reloaded = StudentCNN()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    reloaded.load_state_dict(ckpt["model_state_dict"])
    reloaded_loss, reloaded_acc = evaluate(reloaded, loader, crit, torch.device("cpu"))

    assert abs(reloaded_loss - loss) < 1e-5
    assert abs(reloaded_acc - acc) < 1e-5
