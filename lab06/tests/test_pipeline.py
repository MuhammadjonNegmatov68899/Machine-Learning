"""
Unit tests for YAML pipeline execution and MLflow integration.
"""

import os
import tempfile
import pytest
from lab06.src.pipeline import train_pipeline
from lab06.src.utils import save_yaml_config


def test_pipeline_smoke_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = {
            "seed": 42,
            "dataset": {
                "name": "FashionMNIST",
                "data_dir": "lab03/data",
                "val_split": 5000,
                "batch_size": 256,
                "num_workers": 0,
            },
            "model": {
                "architecture": "CNN",
                "in_channels": 1,
                "c1": 16,
                "c2": 32,
                "num_classes": 10,
            },
            "training": {
                "epochs": 1,
                "device": "cpu",
                "optimizer": {"type": "Adam", "lr": 0.001},
                "scheduler": {"type": "None"},
            },
            "checkpoint": {
                "save_dir": os.path.join(tmpdir, "checkpoints"),
                "save_best_only": True,
                "monitor": "val_acc",
                "mode": "max",
                "filename_prefix": "smoke_model",
            },
            "mlflow": {
                "enabled": True,
                "experiment_name": "test_experiment",
                "run_name": "smoke_run",
                "tracking_uri": os.path.join(tmpdir, "mlruns"),
            },
            "plots": {
                "save_dir": os.path.join(tmpdir, "plots"),
            },
        }

        cfg_path = os.path.join(tmpdir, "test_config.yaml")
        save_yaml_config(cfg, cfg_path)

        res = train_pipeline(cfg_path, override_device="cpu")
        assert "test_loss" in res
        assert "test_acc" in res
        assert res["test_acc"] > 50.0  # CNN easily exceeds 50% after 1 epoch
        assert os.path.exists(res["best_checkpoint_path"])
