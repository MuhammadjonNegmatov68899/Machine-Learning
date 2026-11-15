"""
Unit tests for Gradient Accumulation logic and gradient consistency.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from lab06.src.models import CNN


def test_gradient_accumulation_equivalence():
    """
    Verifies that accumulating gradients over 2 micro-batches of size 4
    produces identical parameter updates to a single mini-batch of size 8
    (when summation/averaging order is identical).
    """
    torch.manual_seed(42)
    x = torch.randn(8, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)

    # Model A: Full Batch of 8
    torch.manual_seed(100)
    model_a = CNN()
    opt_a = optim.SGD(model_a.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()

    opt_a.zero_grad()
    loss_a = crit(model_a(x), y)
    loss_a.backward()
    opt_a.step()

    # Model B: Micro-batches of 4 with accumulation steps 2
    torch.manual_seed(100)
    model_b = CNN()
    opt_b = optim.SGD(model_b.parameters(), lr=0.01)

    opt_b.zero_grad()
    # Micro-batch 1 (first 4)
    loss_b1 = crit(model_b(x[:4]), y[:4]) / 2.0
    loss_b1.backward()

    # Micro-batch 2 (second 4)
    loss_b2 = crit(model_b(x[4:]), y[4:]) / 2.0
    loss_b2.backward()

    opt_b.step()

    # Assert model parameters are essentially identical
    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-5), f"Mismatch in {name_a} after accumulation"


def test_gradient_accumulation_incomplete_final_group():
    """
    Verifies that gradient accumulation with sample-weighted normalization
    for an incomplete final group (e.g. 7 samples split into micro-batches 4 and 3)
    produces identical parameter updates to a single physical batch of 7.
    """
    torch.manual_seed(42)
    x = torch.randn(7, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.long)

    # Model A: Full Batch of 7
    torch.manual_seed(200)
    model_a = CNN()
    opt_a = optim.SGD(model_a.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()

    opt_a.zero_grad()
    loss_a = crit(model_a(x), y)
    loss_a.backward()
    opt_a.step()

    # Model B: Micro-batches of 4 and 3 with exact sample weighting
    torch.manual_seed(200)
    model_b = CNN()
    opt_b = optim.SGD(model_b.parameters(), lr=0.01)

    opt_b.zero_grad()
    # Micro-batch 1: 4 samples out of 7 total in the group
    loss_b1 = crit(model_b(x[:4]), y[:4]) * (4.0 / 7.0)
    loss_b1.backward()

    # Micro-batch 2: 3 samples out of 7 total in the group
    loss_b2 = crit(model_b(x[4:]), y[4:]) * (3.0 / 7.0)
    loss_b2.backward()

    opt_b.step()

    for (name_a, p_a), (name_b, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-5), f"Incomplete group mismatch in {name_a}"
