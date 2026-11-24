"""
End-to-end smoke tests verifying that the full pipeline executes cleanly.
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from lab07.src.models import BaselineCNN, StudentCNN
from lab07.src.pruning import (
    apply_global_unstructured_pruning,
    apply_layerwise_unstructured_pruning,
    apply_structured_filter_pruning,
)
from lab07.src.training import finetune_pruned_model, train_distilled_student


@pytest.fixture
def tiny_dataset():
    """Creates a tiny synthetic Fashion-MNIST dataset for fast smoke testing."""
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=8, shuffle=False)


def test_smoke_global_pruning_and_finetune(tiny_dataset):
    """Smoke test: global pruning followed by 1-epoch fine-tune."""
    model = BaselineCNN()
    pruned_m, masks = apply_global_unstructured_pruning(model, 0.50)
    res = finetune_pruned_model(
        pruned_m,
        masks,
        tiny_dataset,
        tiny_dataset,
        tiny_dataset,
        epochs=1,
        lr=0.001,
        device=torch.device("cpu"),
    )
    assert "post_finetune_test_acc" in res


def test_smoke_structured_pruning_and_finetune(tiny_dataset):
    """Smoke test: structured filter pruning followed by 1-epoch fine-tune."""
    model = BaselineCNN()
    struct_m = apply_structured_filter_pruning(model, 0.25)
    res = finetune_pruned_model(
        struct_m,
        None,
        tiny_dataset,
        tiny_dataset,
        tiny_dataset,
        epochs=1,
        lr=0.001,
        device=torch.device("cpu"),
    )
    assert "post_finetune_test_acc" in res


def test_smoke_distillation(tiny_dataset):
    """Smoke test: knowledge distillation from teacher to student for 1 epoch."""
    teacher = BaselineCNN()
    student = StudentCNN()
    res = train_distilled_student(
        student,
        teacher,
        tiny_dataset,
        tiny_dataset,
        tiny_dataset,
        epochs=1,
        lr=0.001,
        device=torch.device("cpu"),
    )
    assert "test_acc" in res
