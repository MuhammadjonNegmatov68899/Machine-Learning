"""
Unit tests for Checkpoint saving, loading, and state restoration.
"""

import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from lab06.src.models import CNN
from lab06.src.pipeline import save_checkpoint, load_checkpoint


def test_checkpoint_save_and_load_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = os.path.join(tmpdir, "test_chk.pt")

        model1 = CNN()
        optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
        scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=2, gamma=0.5)

        # Perturb weights via dummy step
        x = torch.randn(2, 1, 28, 28)
        y = torch.tensor([0, 1], dtype=torch.long)
        loss = nn.CrossEntropyLoss()(model1(x), y)
        loss.backward()
        optimizer1.step()
        scheduler1.step()

        metrics = {"val_acc": 88.5, "val_loss": 0.35}
        config = {"seed": 42, "lr": 1e-3}

        gen1 = torch.Generator().manual_seed(999)
        # Advance generator state
        torch.randn(10, generator=gen1)

        save_checkpoint(
            chk_path,
            epoch=2,
            model=model1,
            optimizer=optimizer1,
            scheduler=scheduler1,
            metrics=metrics,
            config=config,
            dataloader_generator=gen1,
        )
        assert os.path.exists(chk_path)

        # Restore into a fresh model
        model2 = CNN()
        optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=2, gamma=0.5)
        gen2 = torch.Generator()

        restored_chk = load_checkpoint(
            chk_path,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            restore_rng=True,
            dataloader_generator=gen2,
            weights_only=False,
        )

        assert restored_chk["epoch"] == 2
        assert restored_chk["metrics"]["val_acc"] == 88.5

        # Verify DataLoader generator state produces identical subsequent random numbers
        rand1 = torch.randn(5, generator=gen1)
        rand2 = torch.randn(5, generator=gen2)
        assert torch.equal(rand1, rand2)

        # Verify weights match perfectly
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.equal(p1, p2)

        # Verify optimizer state matches
        sd1 = optimizer1.state_dict()
        sd2 = optimizer2.state_dict()
        assert sd1["param_groups"] == sd2["param_groups"]
        for p_id in sd1["state"]:
            assert p_id in sd2["state"]
            for k in sd1["state"][p_id]:
                v1 = sd1["state"][p_id][k]
                v2 = sd2["state"][p_id][k]
                if isinstance(v1, torch.Tensor):
                    assert torch.equal(v1, v2)
                else:
                    assert v1 == v2
