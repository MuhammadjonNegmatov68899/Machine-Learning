"""
Machine Learning Lab 05: Frameworks and Automatic Differentiation.

Contains:
- Value node and reverse-mode automatic differentiation engine.
- High-level tensor/vector operations and finite-difference gradient verification.
- Scalar neural network modules (Neuron, Layer, MLP) and equivalent PyTorch model.
- Training routines on sklearn.datasets.make_moons with 5-seed comparison.
- PyTorch eager vs compiled/traced benchmarks and tensor contiguity analysis.
"""

from .engine import Value
from .neural_network import Neuron, Layer, MLP, PyTorchMLP
from .operations import (
    sum_values,
    mean_values,
    binary_cross_entropy,
    finite_difference_gradient,
    verify_expression_gradients,
    verify_vector_expression_gradients,
)

__all__ = [
    "Value",
    "Neuron",
    "Layer",
    "MLP",
    "PyTorchMLP",
    "sum_values",
    "mean_values",
    "binary_cross_entropy",
    "finite_difference_gradient",
    "verify_expression_gradients",
    "verify_vector_expression_gradients",
]
