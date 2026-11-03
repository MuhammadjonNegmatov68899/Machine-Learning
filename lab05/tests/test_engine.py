"""
Unit Tests for Task 1 (Value Node), Task 2 (Operations), and Task 3 (Backpropagation).
"""

import math
import pytest
import torch

from src.engine import Value


class TestValueNodeAndOperations:
    """Tests for Value node representation, operators, and forward computations."""

    def test_value_initialization(self):
        v = Value(3.14, label="pi")
        assert v.data == 3.14
        assert v.grad == 0.0
        assert len(v._prev) == 0
        assert v._op == ""
        assert v.label == "pi"
        assert "Value(data=3.140000" in repr(v)

    def test_addition_forward_and_backward(self):
        a = Value(3.0)
        b = Value(4.0)
        c = a + b
        assert c.data == 7.0
        assert c._op == "+"
        assert a in c._prev and b in c._prev

        c.backward()
        assert c.grad == 1.0
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_multiplication_forward_and_backward(self):
        a = Value(3.0)
        b = Value(-4.0)
        c = a * b
        assert c.data == -12.0
        assert c._op == "*"

        c.backward()
        assert a.grad == -4.0
        assert b.grad == 3.0

    def test_power_forward_and_backward(self):
        a = Value(3.0)
        b = a ** 3
        assert b.data == 27.0
        assert b._op == "**3"

        b.backward()
        # d(x^3)/dx = 3 * x^2 = 3 * 9 = 27
        assert a.grad == 27.0

    def test_relu_forward_and_backward(self):
        # Positive branch
        pos = Value(5.0)
        out_pos = pos.relu()
        assert out_pos.data == 5.0
        out_pos.backward()
        assert pos.grad == 1.0

        # Negative branch
        neg = Value(-5.0)
        out_neg = neg.relu()
        assert out_neg.data == 0.0
        out_neg.backward()
        assert neg.grad == 0.0

    def test_negation_and_subtraction(self):
        a = Value(10.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 7.0

        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_division(self):
        a = Value(12.0)
        b = Value(4.0)
        c = a / b
        assert c.data == 3.0

        c.backward()
        # d(a/b)/da = 1/b = 1/4 = 0.25
        # d(a/b)/db = -a / b^2 = -12 / 16 = -0.75
        assert a.grad == 0.25
        assert b.grad == -0.75

    def test_division_by_zero_raises(self):
        a = Value(1.0)
        b = Value(0.0)
        with pytest.raises(ZeroDivisionError):
            _ = a / b

    def test_exp_and_log(self):
        a = Value(2.0)
        b = a.exp()
        assert math.isclose(b.data, math.exp(2.0))
        b.backward()
        assert math.isclose(a.grad, math.exp(2.0))

        c = Value(5.0)
        d = c.log()
        assert math.isclose(d.data, math.log(5.0))
        d.backward()
        assert math.isclose(c.grad, 1.0 / 5.0)

    def test_sigmoid_forward_and_backward(self):
        a = Value(0.0)
        s = a.sigmoid()
        assert math.isclose(s.data, 0.5)

        s.backward()
        # sigma'(0) = 0.5 * (1 - 0.5) = 0.25
        assert math.isclose(a.grad, 0.25)

    def test_tanh_forward_and_backward(self):
        a = Value(0.0)
        t = a.tanh()
        assert math.isclose(t.data, 0.0)

        t.backward()
        # tanh'(0) = 1 - 0 = 1.0
        assert math.isclose(a.grad, 1.0)

    def test_sum_scalar_identity(self):
        a = Value(42.0)
        s = a.sum()
        assert s.data == 42.0
        s.backward()
        assert a.grad == 1.0


class TestReverseModeAutomaticDifferentiation:
    """Tests for Task 3: reverse-mode backpropagation and graph topology."""

    def test_shared_parent_addition(self):
        """Verify gradient accumulation for y = x + x (derivative should be 2)."""
        x = Value(3.0)
        y = x + x
        y.backward()
        assert y.data == 6.0
        assert x.grad == 2.0

    def test_shared_parent_multiplication(self):
        """Verify gradient accumulation for y = x * x (derivative should be 2x)."""
        x = Value(4.0)
        y = x * x
        y.backward()
        assert y.data == 16.0
        assert x.grad == 8.0  # 2 * 4.0

    def test_diamond_graph_topology(self):
        """
        Verify gradient accumulation on a diamond graph:
            z = (x + y) * (x - y) = x^2 - y^2
            dz/dx = 2x
            dz/dy = -2y
        """
        x = Value(5.0)
        y = Value(3.0)
        a = x + y  # 8.0
        b = x - y  # 2.0
        z = a * b  # 16.0
        z.backward()

        assert z.data == 16.0
        assert x.grad == 10.0  # 2 * 5.0
        assert y.grad == -6.0  # -2 * 3.0

    def test_complex_dag_vs_pytorch(self):
        """
        Compare complex multi-branch DAG against PyTorch:
            f(x, y) = ((x * y + exp(x)) ** 2) / (x + y)
        """
        x_val, y_val = 1.5, 2.5

        # Custom
        x_c = Value(x_val)
        y_c = Value(y_val)
        num = (x_c * y_c + x_c.exp()) ** 2
        den = x_c + y_c
        f_c = num / den
        f_c.backward()

        # PyTorch
        x_t = torch.tensor(x_val, dtype=torch.float64, requires_grad=True)
        y_t = torch.tensor(y_val, dtype=torch.float64, requires_grad=True)
        num_t = (x_t * y_t + torch.exp(x_t)) ** 2
        den_t = x_t + y_t
        f_t = num_t / den_t
        f_t.backward()

        assert math.isclose(f_c.data, f_t.item(), rel_tol=1e-9)
        assert math.isclose(x_c.grad, x_t.grad.item(), rel_tol=1e-9)
        assert math.isclose(y_c.grad, y_t.grad.item(), rel_tol=1e-9)

    def test_zero_grad(self):
        """Test recursive zero_grad resets all ancestor node gradients."""
        x = Value(2.0)
        y = Value(3.0)
        z = x * y + y
        z.backward()

        assert z.grad != 0.0
        assert x.grad != 0.0
        assert y.grad != 0.0

        z.zero_grad()
        assert z.grad == 0.0
        assert x.grad == 0.0
        assert y.grad == 0.0
