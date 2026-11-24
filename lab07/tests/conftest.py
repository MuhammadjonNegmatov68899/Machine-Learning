"""
Test suite configuration and fixtures for Lab 07.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from lab07.src.utils import set_seed
from lab07.src.models import BaselineCNN, StudentCNN


@pytest.fixture(autouse=True)
def run_around_tests():
    """Sets reproducible seed before each test."""
    set_seed(42)
    yield


@pytest.fixture
def baseline_model():
    """Returns an initialized BaselineCNN model."""
    return BaselineCNN()


@pytest.fixture
def student_model():
    """Returns an initialized StudentCNN model."""
    return StudentCNN()


@pytest.fixture
def dummy_input():
    """Returns a batch of dummy Fashion-MNIST tensors."""
    return torch.randn(4, 1, 28, 28)
