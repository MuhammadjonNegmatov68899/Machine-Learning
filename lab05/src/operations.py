"""
High-Level Operations, Functional Primitives, and Gradient Verification.

Provides:
- Reduction operations: sum_values, mean_values
- Loss functions: binary_cross_entropy
- Numerical differentiation via central finite differences
- Verification routines comparing Custom Engine against PyTorch and finite differences
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Union

import pandas as pd
import torch

try:
    from src.engine import Value
except ImportError:
    from .engine import Value


def sum_values(values: Iterable[Value]) -> Value:
    """
    Sum an iterable of scalar Value nodes into a single scalar Value node.

    Forward: sum_i (v_i.data)
    Backward: d(out)/d(v_i) = 1.0 * out.grad
    """
    val_list = list(values)
    if not val_list:
        return Value(0.0)

    total_data = sum(v.data for v in val_list)
    out = Value(total_data, tuple(val_list), "sum_all")

    def _backward() -> None:
        for v in val_list:
            v.grad += 1.0 * out.grad

    out._backward = _backward
    return out


def mean_values(values: Iterable[Value]) -> Value:
    """
    Compute arithmetic mean of an iterable of scalar Value nodes.

    Forward: (1 / N) * sum_i (v_i.data)
    Backward: d(out)/d(v_i) = (1.0 / N) * out.grad
    """
    val_list = list(values)
    n = len(val_list)
    if n == 0:
        raise ValueError("Cannot compute mean of empty sequence.")

    total_data = sum(v.data for v in val_list)
    out = Value(total_data / n, tuple(val_list), "mean")

    def _backward() -> None:
        scale = 1.0 / n
        for v in val_list:
            v.grad += scale * out.grad

    out._backward = _backward
    return out


def binary_cross_entropy(
    y_pred: Sequence[Value],
    y_true: Sequence[Union[int, float]],
    eps: float = 1e-15,
) -> Value:
    """
    Compute mean Binary Cross-Entropy loss over scalar Value predictions
    and ground-truth binary targets (0 or 1).

    Loss = - (1 / N) * sum_i [ y_i * ln(p_i + eps) + (1 - y_i) * ln(1 - p_i + eps) ]

    Args:
        y_pred: Sequence of predicted probabilities (Value objects, in [0, 1]).
        y_true: Sequence of target labels (0.0 or 1.0).
        eps: Small numerical constant to prevent log(0).

    Returns:
        Scalar Value node representing mean BCE loss.
    """
    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Length mismatch: {len(y_pred)} predictions vs {len(y_true)} targets."
        )

    losses: List[Value] = []
    for p, y in zip(y_pred, y_true):
        y_val = float(y)
        # Numerically stabilized loss computation within the Value graph
        # y * ln(p) + (1 - y) * ln(1 - p)
        # Using p.log(eps) for safe differentiation
        term_pos = Value(y_val) * p.log(eps=eps)
        term_neg = Value(1.0 - y_val) * (Value(1.0) - p).log(eps=eps)
        loss_sample = -(term_pos + term_neg)
        losses.append(loss_sample)

    return mean_values(losses)


# =============================================================================
# Task 4: Numerical Finite Differences & Verification
# =============================================================================

def finite_difference_gradient(
    func: Callable[[Dict[str, float]], float],
    inputs: Dict[str, float],
    target_var: str,
    eps: float = 1e-6,
) -> float:
    """
    Compute central finite-difference numerical derivative with respect to target_var:
        f'(x) ~= [f(x + eps) - f(x - eps)] / (2 * eps)

    Truncation error: O(eps^2).
    """
    inputs_plus = dict(inputs)
    inputs_minus = dict(inputs)

    inputs_plus[target_var] = inputs[target_var] + eps
    inputs_minus[target_var] = inputs[target_var] - eps

    f_plus = func(inputs_plus)
    f_minus = func(inputs_minus)

    return (f_plus - f_minus) / (2.0 * eps)


def verify_expression_gradients(
    a_val: float = 2.0,
    b_val: float = -3.0,
    c_val: float = 10.0,
    eps: float = 1e-6,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Verify the custom automatic differentiation engine against:
    1. PyTorch autograd.
    2. Central finite-difference numerical gradients.

    Evaluates the mandatory expression:
        ((a * b + c) ** 2).relu().sum()

    Args:
        a_val: Initial scalar value for input variable a.
        b_val: Initial scalar value for input variable b.
        c_val: Initial scalar value for input variable c.
        eps: Step size for numerical finite differences.
        tol: Tolerance threshold (strictly < 1e-6).

    Returns:
        DataFrame containing variable names, Custom gradients, PyTorch gradients,
        Numerical gradients, absolute errors, and Pass/Fail verification status.
    """
    inputs = {"a": float(a_val), "b": float(b_val), "c": float(c_val)}

    # 1. Custom Engine
    a_custom = Value(inputs["a"], label="a")
    b_custom = Value(inputs["b"], label="b")
    c_custom = Value(inputs["c"], label="c")
    out_custom = ((a_custom * b_custom + c_custom) ** 2).relu().sum()
    out_custom.backward()

    custom_grads = {
        "a": a_custom.grad,
        "b": b_custom.grad,
        "c": c_custom.grad,
    }

    # 2. PyTorch Autograd (double precision for ground truth)
    a_torch = torch.tensor(inputs["a"], dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(inputs["b"], dtype=torch.float64, requires_grad=True)
    c_torch = torch.tensor(inputs["c"], dtype=torch.float64, requires_grad=True)
    out_torch = ((a_torch * b_torch + c_torch) ** 2).relu().sum()
    out_torch.backward()

    torch_grads = {
        "a": float(a_torch.grad.item()),
        "b": float(b_torch.grad.item()),
        "c": float(c_torch.grad.item()),
    }

    # 3. Numerical Finite-Difference
    def eval_math(params: Dict[str, float]) -> float:
        inner = params["a"] * params["b"] + params["c"]
        squared = inner ** 2
        rel = max(0.0, squared)
        return float(rel)

    num_grads = {
        var: finite_difference_gradient(eval_math, inputs, var, eps=eps)
        for var in ("a", "b", "c")
    }

    # Assemble comparison table
    records = []
    for var in ("a", "b", "c"):
        c_g = custom_grads[var]
        t_g = torch_grads[var]
        n_g = num_grads[var]
        err_torch = abs(c_g - t_g)
        err_num = abs(c_g - n_g)
        max_err = max(err_torch, err_num)
        passed = max_err < tol

        records.append({
            "variable": var,
            "input_value": inputs[var],
            "custom_grad": c_g,
            "torch_grad": t_g,
            "num_grad": n_g,
            "abs_err_vs_torch": err_torch,
            "abs_err_vs_num": err_num,
            "max_abs_error": max_err,
            "tolerance": tol,
            "status": "PASSED" if passed else "FAILED",
        })

    return pd.DataFrame(records)


def verify_vector_expression_gradients(
    a_vec: Sequence[float],
    b_vec: Sequence[float],
    c_vec: Sequence[float],
    eps: float = 1e-6,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Verify gradient consistency across vector inputs for:
        sum_i [ ((a_i * b_i + c_i) ** 2).relu() ]
    """
    n = len(a_vec)
    assert len(b_vec) == n and len(c_vec) == n, "Vector lengths must match."

    # 1. Custom Engine
    a_nodes = [Value(x, label=f"a_{i}") for i, x in enumerate(a_vec)]
    b_nodes = [Value(x, label=f"b_{i}") for i, x in enumerate(b_vec)]
    c_nodes = [Value(x, label=f"c_{i}") for i, x in enumerate(c_vec)]

    elem_outputs = [
        ((a_i * b_i + c_i) ** 2).relu()
        for a_i, b_i, c_i in zip(a_nodes, b_nodes, c_nodes)
    ]
    out_custom = sum_values(elem_outputs)
    out_custom.backward()

    # 2. PyTorch
    a_torch = torch.tensor(a_vec, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(b_vec, dtype=torch.float64, requires_grad=True)
    c_torch = torch.tensor(c_vec, dtype=torch.float64, requires_grad=True)
    out_torch = ((a_torch * b_torch + c_torch) ** 2).relu().sum()
    out_torch.backward()

    records = []
    for i in range(n):
        # Numerical finite difference for element i
        def eval_math(var_name: str, delta: float) -> float:
            total = 0.0
            for k in range(n):
                ak = a_vec[k] + (delta if var_name == f"a_{k}" else 0.0)
                bk = b_vec[k] + (delta if var_name == f"b_{k}" else 0.0)
                ck = c_vec[k] + (delta if var_name == f"c_{k}" else 0.0)
                total += max(0.0, (ak * bk + ck) ** 2)
            return total

        for var_prefix, custom_node, torch_tensor, val in (
            ("a", a_nodes[i], a_torch.grad[i], a_vec[i]),
            ("b", b_nodes[i], b_torch.grad[i], b_vec[i]),
            ("c", c_nodes[i], c_torch.grad[i], c_vec[i]),
        ):
            v_name = f"{var_prefix}_{i}"
            num_g = (
                eval_math(v_name, eps) - eval_math(v_name, -eps)
            ) / (2.0 * eps)
            c_g = custom_node.grad
            t_g = float(torch_tensor.item())

            err_t = abs(c_g - t_g)
            err_n = abs(c_g - num_g)
            max_err = max(err_t, err_n)
            passed = max_err < tol

            records.append({
                "variable": v_name,
                "input_value": val,
                "custom_grad": c_g,
                "torch_grad": t_g,
                "num_grad": num_g,
                "abs_err_vs_torch": err_t,
                "abs_err_vs_num": err_n,
                "max_abs_error": max_err,
                "tolerance": tol,
                "status": "PASSED" if passed else "FAILED",
            })

    return pd.DataFrame(records)
