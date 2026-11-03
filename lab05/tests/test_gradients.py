"""
Unit Tests for Task 4: Gradient Verification.

Verifies custom automatic differentiation against:
1. PyTorch Autograd.
2. Central Finite-Difference Numerical Differentiation.

Strict Requirement: Maximum absolute error must be strictly less than 1e-6.
Target Expression: ((a * b + c) ** 2).relu().sum()
"""

import pytest
import torch

from src.engine import Value
from src.operations import (
    finite_difference_gradient,
    verify_expression_gradients,
    verify_vector_expression_gradients,
)


class TestGradientVerification:
    """Rigorous verification of the expression ((a * b + c) ** 2).relu().sum()."""

    @pytest.mark.parametrize(
        "a, b, c",
        [
            (2.0, -3.0, 10.0),    # Primary university assignment test case
            (1.5, 2.5, -4.0),     # Negative inner term
            (-3.0, -2.0, -1.0),   # All negative/positive mixture
            (0.5, 4.0, 0.0),      # Zero intercept c
            (2.0, 3.0, 4.0),      # Moderate positive magnitude
        ],
    )
    def test_gradient_verification_strict_tolerance(self, a, b, c):
        """
        Verify that maximum absolute error against both PyTorch and
        numerical gradients is strictly less than 1e-6.
        """
        df = verify_expression_gradients(a_val=a, b_val=b, c_val=c, eps=1e-6, tol=1e-6)

        for _, row in df.iterrows():
            var_name = row["variable"]
            err_torch = row["abs_err_vs_torch"]
            err_num = row["abs_err_vs_num"]
            max_err = row["max_abs_error"]
            status = row["status"]

            # Assertion against PyTorch
            assert err_torch < 1e-6, (
                f"Variable {var_name}: error vs PyTorch {err_torch:.3e} exceeds 1e-6."
            )

            # Assertion against Finite Difference
            assert err_num < 1e-6, (
                f"Variable {var_name}: error vs Numerical {err_num:.3e} exceeds 1e-6."
            )

            # Overall pass status
            assert max_err < 1e-6
            assert status == "PASSED"

    def test_analytical_manual_derivatives(self):
        """
        Verify that gradients match exact analytical calculus for:
            u = a * b + c = 2(-3) + 10 = 4.0
            v = u^2 = 16.0
            w = relu(v) = 16.0
            sum = 16.0
            dz/da = 2 * u * b = 2 * 4 * (-3) = -24.0
            dz/db = 2 * u * a = 2 * 4 * 2 = 16.0
            dz/dc = 2 * u * 1 = 2 * 4 * 1 = 8.0
        """
        a = Value(2.0)
        b = Value(-3.0)
        c = Value(10.0)

        out = ((a * b + c) ** 2).relu().sum()
        out.backward()

        assert out.data == 16.0
        assert a.grad == -24.0
        assert b.grad == 16.0
        assert c.grad == 8.0

    def test_vector_gradient_verification(self):
        """Verify vector input gradients across 5 dimensions."""
        a_vec = [1.5, -2.0, 3.0, -0.5, 2.2]
        b_vec = [-1.0, 2.5, 0.5, -3.0, 1.2]
        c_vec = [0.5, 1.0, -2.0, 4.0, -1.5]

        df = verify_vector_expression_gradients(a_vec, b_vec, c_vec, eps=1e-6, tol=1e-6)

        max_err = df["max_abs_error"].max()
        assert max_err < 1e-6, f"Max vector error {max_err:.3e} exceeds 1e-6"
        assert all(status == "PASSED" for status in df["status"])
