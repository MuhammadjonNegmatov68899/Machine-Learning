"""
Training Routines and Experimental Comparisons on make_moons.

Implements Task 5:
- Data generation and leakage-free normalization using make_moons.
- Custom engine neural network training loop.
- PyTorch neural network training loop.
- 5-seed reproducible comparison measuring training loss, accuracy, and latency.
- Computation of median metrics across runs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from src.neural_network import MLP, PyTorchMLP
    from src.operations import binary_cross_entropy
except ImportError:
    from .neural_network import MLP, PyTorchMLP
    from .operations import binary_cross_entropy


def prepare_moons_data(
    n_samples: int = 200,
    noise: float = 0.20,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Generate and partition the make_moons dataset.

    Applies a leakage-free protocol where the StandardScaler is fitted
    strictly on the training partition and applied to the test partition.

    Returns:
        X_train, X_test, y_train, y_test, fitted_scaler
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def compute_accuracy(preds: Sequence[float], targets: Sequence[int | float]) -> float:
    """Compute binary classification accuracy given predicted probabilities."""
    correct = sum((1 if p >= 0.5 else 0) == int(t) for p, t in zip(preds, targets))
    return float(correct) / len(targets)


def train_custom_engine(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    lr: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a 2 -> 8 -> 1 MLP using the custom scalar automatic differentiation engine.

    Measures training latency using high-precision time.perf_counter().
    """
    model = MLP(nin=2, nouts=[8, 1], activations=["sigmoid", "sigmoid"], seed=seed)

    X_tr_list = X_train.tolist()
    y_tr_list = y_train.tolist()
    X_te_list = X_test.tolist()
    y_te_list = y_test.tolist()

    history_loss: List[float] = []
    history_train_acc: List[float] = []
    history_test_acc: List[float] = []

    start_time = time.perf_counter()

    for epoch in range(epochs):
        # 1. Forward pass
        preds = [model(x) for x in X_tr_list]
        loss = binary_cross_entropy(preds, y_tr_list)

        # 2. Backward pass (reverse-mode AD)
        model.zero_grad()
        loss.backward()

        # 3. Gradient descent parameter update
        for p in model.parameters():
            p.data -= lr * p.grad

        # 4. Metrics evaluation
        train_probs = [p.data for p in preds]
        train_acc = compute_accuracy(train_probs, y_tr_list)
        history_loss.append(loss.data)
        history_train_acc.append(train_acc)

        # Periodic test evaluation
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            test_probs = [model(x).data for x in X_te_list]
            history_test_acc.append(compute_accuracy(test_probs, y_te_list))

    total_time = time.perf_counter() - start_time

    # Final evaluation
    final_test_preds = [model(x).data for x in X_te_list]
    final_train_preds = [model(x).data for x in X_tr_list]

    # Evaluate test loss
    test_loss_node = binary_cross_entropy(
        [model(x) for x in X_te_list], y_te_list
    )

    return {
        "engine": "custom",
        "seed": seed,
        "epochs": epochs,
        "train_time_sec": total_time,
        "final_train_loss": float(history_loss[-1]),
        "final_test_loss": float(test_loss_node.data),
        "final_train_acc": compute_accuracy(final_train_preds, y_tr_list),
        "final_test_acc": compute_accuracy(final_test_preds, y_te_list),
        "history_loss": history_loss,
        "history_train_acc": history_train_acc,
        "model": model,
    }


def train_pytorch_engine(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    lr: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train an equivalent 2 -> 8 -> 1 MLP using PyTorch.

    Uses torch.optim.SGD with identical learning rate and BCELoss.
    """
    torch.manual_seed(seed)
    model = PyTorchMLP(
        nin=2,
        hidden_dim=8,
        nout=1,
        hidden_activation="sigmoid",
        seed=seed,
    )

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.float32)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history_loss: List[float] = []
    history_train_acc: List[float] = []
    history_test_acc: List[float] = []

    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(X_tr_t).squeeze(-1)
        loss = criterion(preds, y_tr_t)

        loss.backward()
        optimizer.step()

        train_probs = preds.detach().cpu().numpy()
        train_acc = compute_accuracy(train_probs.tolist(), y_train.tolist())

        history_loss.append(float(loss.item()))
        history_train_acc.append(train_acc)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_preds = model(X_te_t).squeeze(-1)
                test_acc = compute_accuracy(
                    test_preds.cpu().numpy().tolist(), y_test.tolist()
                )
                history_test_acc.append(test_acc)

    total_time = time.perf_counter() - start_time

    model.eval()
    with torch.no_grad():
        final_test_preds = model(X_te_t).squeeze(-1)
        final_test_loss = criterion(final_test_preds, y_te_t).item()
        final_train_preds = model(X_tr_t).squeeze(-1)

    return {
        "engine": "pytorch",
        "seed": seed,
        "epochs": epochs,
        "train_time_sec": total_time,
        "final_train_loss": float(history_loss[-1]),
        "final_test_loss": float(final_test_loss),
        "final_train_acc": compute_accuracy(
            final_train_preds.cpu().numpy().tolist(), y_train.tolist()
        ),
        "final_test_acc": compute_accuracy(
            final_test_preds.cpu().numpy().tolist(), y_test.tolist()
        ),
        "history_loss": history_loss,
        "history_train_acc": history_train_acc,
        "model": model,
    }


def run_5_fold_comparison(
    seeds: Sequence[int] = (42, 1337, 2024, 7, 999),
    epochs: int = 100,
    lr: float = 1.0,
    n_samples: int = 200,
    noise: float = 0.20,
    test_size: float = 0.25,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute 5 reproducible runs comparing Custom Engine vs PyTorch.

    Returns:
        DataFrame of run-level results plus summary median rows,
        and dictionary containing loss history curves for plotting.
    """
    X_train, X_test, y_train, y_test, _ = prepare_moons_data(
        n_samples=n_samples, noise=noise, test_size=test_size, random_state=42
    )

    records: List[Dict[str, Any]] = []
    custom_histories: List[List[float]] = []
    torch_histories: List[List[float]] = []

    for run_idx, seed in enumerate(seeds, start=1):
        # 1. Custom Engine
        custom_res = train_custom_engine(
            X_train, y_train, X_test, y_test, epochs=epochs, lr=lr, seed=seed
        )
        custom_histories.append(custom_res["history_loss"])

        # 2. PyTorch
        torch_res = train_pytorch_engine(
            X_train, y_train, X_test, y_test, epochs=epochs, lr=lr, seed=seed
        )
        torch_histories.append(torch_res["history_loss"])

        records.append({
            "run_id": run_idx,
            "seed": seed,
            "engine": "custom",
            "epochs": epochs,
            "train_loss": custom_res["final_train_loss"],
            "test_loss": custom_res["final_test_loss"],
            "train_acc": custom_res["final_train_acc"],
            "test_acc": custom_res["final_test_acc"],
            "train_time_sec": custom_res["train_time_sec"],
        })

        records.append({
            "run_id": run_idx,
            "seed": seed,
            "engine": "pytorch",
            "epochs": epochs,
            "train_loss": torch_res["final_train_loss"],
            "test_loss": torch_res["final_test_loss"],
            "train_acc": torch_res["final_train_acc"],
            "test_acc": torch_res["final_test_acc"],
            "train_time_sec": torch_res["train_time_sec"],
        })

    df = pd.DataFrame(records)

    # Compute median across 5 runs for each engine
    custom_sub = df[df["engine"] == "custom"]
    torch_sub = df[df["engine"] == "pytorch"]

    median_custom = {
        "run_id": "MEDIAN",
        "seed": "All (5 seeds)",
        "engine": "custom",
        "epochs": epochs,
        "train_loss": float(custom_sub["train_loss"].median()),
        "test_loss": float(custom_sub["test_loss"].median()),
        "train_acc": float(custom_sub["train_acc"].median()),
        "test_acc": float(custom_sub["test_acc"].median()),
        "train_time_sec": float(custom_sub["train_time_sec"].median()),
    }

    median_torch = {
        "run_id": "MEDIAN",
        "seed": "All (5 seeds)",
        "engine": "pytorch",
        "epochs": epochs,
        "train_loss": float(torch_sub["train_loss"].median()),
        "test_loss": float(torch_sub["test_loss"].median()),
        "train_acc": float(torch_sub["train_acc"].median()),
        "test_acc": float(torch_sub["test_acc"].median()),
        "train_time_sec": float(torch_sub["train_time_sec"].median()),
    }

    summary_df = pd.concat([df, pd.DataFrame([median_custom, median_torch])], ignore_index=True)

    plot_data = {
        "custom_loss_histories": custom_histories,
        "torch_loss_histories": torch_histories,
        "custom_times": custom_sub["train_time_sec"].tolist(),
        "torch_times": torch_sub["train_time_sec"].tolist(),
        "seeds": list(seeds),
    }

    return summary_df, plot_data
