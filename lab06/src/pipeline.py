"""
YAML-Driven Training Pipeline with Checkpointing and MLflow Experiment Tracking.
Implements Task 1 requirements.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import os
import time
import random
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

try:
    import mlflow
except ImportError:
    mlflow = None

from .models import CNN, get_model
from .dataset import get_fashion_mnist_dataloaders
from .utils import (
    set_seed,
    get_device,
    load_yaml_config,
    flatten_dict,
    setup_plot_style,
)


def get_optimizer(model: nn.Module, opt_cfg: Dict[str, Any]) -> optim.Optimizer:
    """Instantiates optimizer based on configuration."""
    opt_type = opt_cfg.get("type", "Adam").strip()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if opt_type.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type.lower() == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}. Expected 'Adam' or 'SGD'.")


def get_scheduler(
    optimizer: optim.Optimizer,
    sched_cfg: Optional[Dict[str, Any]],
) -> Optional[Any]:
    """Instantiates learning rate scheduler based on configuration."""
    if not sched_cfg:
        return None

    sched_type = sched_cfg.get("type", "None").strip()
    if sched_type.lower() in ("none", "null", ""):
        return None
    elif sched_type.lower() == "steplr":
        step_size = int(sched_cfg.get("step_size", 3))
        gamma = float(sched_cfg.get("gamma", 0.5))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_type.lower() == "cosineannealinglr":
        t_max = int(sched_cfg.get("T_max", 10))
        eta_min = float(sched_cfg.get("eta_min", 1e-5))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif sched_type.lower() == "reducelronplateau":
        mode = sched_cfg.get("mode", "min")
        factor = float(sched_cfg.get("factor", 0.5))
        patience = int(sched_cfg.get("patience", 1))
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}.")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates loss and accuracy on a given DataLoader.
    
    Returns:
        (average_loss, accuracy_percentage)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.inference_mode():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

    if device.type == "mps":
        torch.mps.synchronize()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = 100.0 * (correct / total_samples) if total_samples > 0 else 0.0
    return avg_loss, accuracy


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    metrics: Dict[str, float],
    config: Dict[str, Any],
    dataloader_generator: Optional[torch.Generator] = None,
) -> None:
    """
    Saves a comprehensive training checkpoint containing model, optimizer,
    scheduler states, RNG states (including DataLoader generator), and training metadata.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rng_state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    if dataloader_generator is not None:
        rng_state["dataloader_generator"] = dataloader_generator.get_state()

    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
        "rng_state": rng_state,
        "timestamp": time.time(),
    }
    torch.save(checkpoint_dict, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    restore_rng: bool = True,
    device: Optional[torch.device] = None,
    dataloader_generator: Optional[torch.Generator] = None,
    weights_only: bool = False,
) -> Dict[str, Any]:
    """
    Restores model, optimizer, scheduler, and RNG states from checkpoint.
    Compatible with PyTorch 2.6+ where weights_only defaults to True.
    """
    load_kwargs: Dict[str, Any] = {"map_location": device or "cpu"}
    # Explicitly set weights_only=False for trusted local checkpoints containing
    # full dictionary state (optimizer state, scheduler state, RNG tuples, config).
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = weights_only

    checkpoint = torch.load(path, **load_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if restore_rng and "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]
        if "torch" in rng and rng["torch"] is not None:
            torch_state = rng["torch"]
            if isinstance(torch_state, torch.Tensor):
                torch_state = torch_state.to(torch.uint8).cpu()
            torch.set_rng_state(torch_state)
        if "numpy" in rng and rng["numpy"] is not None:
            np.random.set_state(rng["numpy"])
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])
        if torch.cuda.is_available() and "cuda" in rng and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if dataloader_generator is not None and "dataloader_generator" in rng and rng["dataloader_generator"] is not None:
            dl_gen_state = rng["dataloader_generator"]
            if isinstance(dl_gen_state, torch.Tensor):
                dl_gen_state = dl_gen_state.to(torch.uint8).cpu()
            dataloader_generator.set_state(dl_gen_state)

    return checkpoint


def plot_training_curves(history: Dict[str, list], save_path: str) -> None:
    """Generates and saves dual-panel loss and accuracy training curve plots."""
    setup_plot_style()
    epochs = history["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], marker="o", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], marker="s", label="Val Loss", color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], marker="o", label="Train Acc", color="#2ca02c")
    ax2.plot(epochs, history["val_acc"], marker="s", label="Val Acc", color="#d62728")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def train_pipeline(
    config_or_path: Any,
    override_device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the reproducible YAML-driven training pipeline.
    
    Args:
        config_or_path: Path to YAML config file, or dictionary of config.
        override_device: Optional device string to override config ("mps", "cpu", "cuda").

    Returns:
        Dictionary with trained model, training history, test metrics, and checkpoint paths.
    """
    if isinstance(config_or_path, (str, Path)):
        config_path = str(config_or_path)
        config = load_yaml_config(config_path)
    else:
        config = config_or_path
        config_path = None

    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Hardware Device
    device_pref = override_device or config.get("training", {}).get("device", "auto")
    device = get_device(device_pref)
    print(f"[Pipeline] Using device: {device} (requested: {device_pref})")

    # Dataset & DataLoaders
    dataset_cfg = config.get("dataset", {})
    train_loader, val_loader, test_loader = get_fashion_mnist_dataloaders(
        data_dir=dataset_cfg.get("data_dir", "lab03/data"),
        batch_size=int(dataset_cfg.get("batch_size", 128)),
        seed=seed,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=bool(dataset_cfg.get("pin_memory", False)),
        val_split=int(dataset_cfg.get("val_split", 5000)),
    )

    # Model & Criterion
    model = get_model(config).to(device)
    criterion = nn.CrossEntropyLoss()

    # Training parameters
    train_cfg = config.get("training", {})
    epochs = int(train_cfg.get("epochs", 5))
    optimizer = get_optimizer(model, train_cfg.get("optimizer", {}))
    scheduler = get_scheduler(optimizer, train_cfg.get("scheduler", {}))

    # Checkpoint configuration
    chk_cfg = config.get("checkpoint", {})
    save_dir = chk_cfg.get("save_dir", "lab06/checkpoints")
    save_best_only = chk_cfg.get("save_best_only", True)
    monitor_metric = chk_cfg.get("monitor", "val_acc")
    mode = chk_cfg.get("mode", "max")
    filename_prefix = chk_cfg.get("filename_prefix", "best_model")
    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, f"{filename_prefix}.pt")

    # MLflow Setup
    mlflow_cfg = config.get("mlflow", {})
    use_mlflow = bool(mlflow_cfg.get("enabled", True)) and (mlflow is not None)
    active_run = None

    if use_mlflow:
        tracking_uri = mlflow_cfg.get("tracking_uri", "lab06/mlruns")
        os.makedirs(tracking_uri, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_uri)}")
        exp_name = mlflow_cfg.get("experiment_name", "lab06_training_pipeline")
        mlflow.set_experiment(exp_name)
        run_name = mlflow_cfg.get("run_name", "training_run")
        active_run = mlflow.start_run(run_name=run_name)
        # Log flattened parameters
        flat_params = flatten_dict(config)
        mlflow.log_params(flat_params)
        print(f"[MLflow] Started run '{run_name}' in experiment '{exp_name}' (ID: {active_run.info.run_id})")

    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time_sec": [],
        "throughput": [],
        "lr": [],
    }

    best_val_score = -float("inf") if mode == "max" else float("inf")

    print(f"\n[Pipeline] Starting {epochs} epochs training of {model.__class__.__name__} ({model.count_parameters():,} params)...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train_samples = 0
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total_train_samples += batch_size

        if device.type == "mps":
            torch.mps.synchronize()

        epoch_time = time.time() - t0
        train_loss = running_loss / total_train_samples
        train_acc = 100.0 * (running_correct / total_train_samples)
        throughput = total_train_samples / epoch_time if epoch_time > 0 else 0.0

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Update history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time_sec"].append(epoch_time)
        history["throughput"].append(throughput)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s ({throughput:.1f} smp/s)"
        )

        # MLflow epoch metrics
        if use_mlflow:
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time_sec": epoch_time,
                    "throughput_samples_per_sec": throughput,
                    "lr": current_lr,
                },
                step=epoch,
            )

        # Checkpoint evaluation
        current_score = val_acc if monitor_metric == "val_acc" else val_loss
        is_best = (current_score > best_val_score) if mode == "max" else (current_score < best_val_score)

        if is_best:
            best_val_score = current_score
            save_checkpoint(
                best_checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={"val_loss": val_loss, "val_acc": val_acc, "train_loss": train_loss},
                config=config,
            )
            print(f"  -> Checkpoint saved (New best {monitor_metric}: {best_val_score:.4f})")

        if not save_best_only:
            epoch_chk_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(
                epoch_chk_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={"val_loss": val_loss, "val_acc": val_acc},
                config=config,
            )

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n[Pipeline] Final Test Evaluation -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")

    # Generate training curves plot
    plot_path = os.path.join(
        config.get("plots", {}).get("save_dir", "lab06/plots"),
        "training_curves.png",
    )
    plot_training_curves(history, plot_path)

    # MLflow Final Logging & Artifacts
    if use_mlflow:
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_val_score": best_val_score,
        })
        if config_path and os.path.exists(config_path):
            mlflow.log_artifact(config_path, artifact_path="configs")
        if os.path.exists(best_checkpoint_path):
            mlflow.log_artifact(best_checkpoint_path, artifact_path="checkpoints")
        if os.path.exists(plot_path):
            mlflow.log_artifact(plot_path, artifact_path="plots")
        mlflow.end_run()
        print(f"[MLflow] Run completed and logged artifacts.")

    return {
        "model": model,
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_val_score": best_val_score,
        "best_checkpoint_path": best_checkpoint_path,
        "plot_path": plot_path,
    }
