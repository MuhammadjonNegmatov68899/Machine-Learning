"""
Unit tests for Lab 6 CNN model architecture.
"""

import pytest
import torch
from lab06.src.models import CNN, get_model


def test_cnn_initialization_and_parameter_count():
    model = CNN(in_channels=1, c1=16, c2=32, num_classes=10)
    # Expected: Conv1: 160 + Conv2: 4,640 + Linear: 15,690 = 20,490
    assert model.count_parameters() == 20490


def test_cnn_forward_output_shape():
    model = CNN()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    assert out.shape == (8, 10)


def test_cnn_backward_gradient_flow():
    model = CNN()
    x = torch.randn(4, 1, 28, 28)
    target = torch.tensor([1, 3, 5, 7], dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"


def test_get_model_from_config():
    cfg = {
        "model": {
            "in_channels": 1,
            "c1": 16,
            "c2": 32,
            "num_classes": 10,
        }
    }
    model = get_model(cfg)
    assert isinstance(model, CNN)
    assert model.count_parameters() == 20490
