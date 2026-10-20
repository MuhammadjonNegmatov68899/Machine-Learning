import os
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ============================================================
# 1. DATA PREPARATION
# ============================================================

def load_data():
    """Load, normalize, and split the handwritten digits dataset."""

    X, y = load_digits(return_X_y=True)

    # Normalize pixel values from [0, 16] to [0, 1]
    X = X.astype(np.float64) / 16.0

    # One-hot encode labels
    y_one_hot = np.eye(10)[y]

    # Stratified 70/30 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_one_hot,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# 2. ACTIVATION FUNCTIONS AND LOSS
# ============================================================

def relu(Z):
    """Rectified Linear Unit activation."""
    return np.maximum(0, Z)


def softmax(Z):
    """Numerically stable softmax."""
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)

    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    """Calculate mean categorical cross-entropy loss."""

    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0)

    return -np.mean(
        np.sum(y_true * np.log(y_pred), axis=1)
    )


# ============================================================
# 3. NEURAL NETWORK WITH TWO HIDDEN LAYERS
# ============================================================

class NeuralNetwork:

    def __init__(
        self,
        input_size=64,
        hidden_size=16,
        output_size=10,
        seed=42,
    ):
        rng = np.random.default_rng(seed)

        self.W1 = rng.normal(
            0, 0.01, (input_size, hidden_size)
        )
        self.b1 = np.zeros(hidden_size)

        self.W2 = rng.normal(
            0, 0.01, (hidden_size, hidden_size)
        )
        self.b2 = np.zeros(hidden_size)

        self.W3 = rng.normal(
            0, 0.01, (hidden_size, output_size)
        )
        self.b3 = np.zeros(output_size)

    # --------------------------------------------------------
    # Forward propagation
    # --------------------------------------------------------

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)

        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)

        Z3 = A2 @ self.W3 + self.b3
        A3 = softmax(Z3)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
            "Z3": Z3,
            "A3": A3,
        }

        return A3, cache

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    def backward(self, y_true, cache):
        X = cache["X"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        A3 = cache["A3"]
        Z1 = cache["Z1"]
        Z2 = cache["Z2"]

        m = X.shape[0]

        # Output layer: softmax + cross-entropy
        dZ3 = (A3 - y_true) / m
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0)

        # Second hidden layer
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * (Z2 > 0)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)

        # First hidden layer
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }

    # --------------------------------------------------------
    # Update parameters using gradient descent
    # --------------------------------------------------------

    def update_parameters(self, gradients, learning_rate):
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]

        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

        self.W3 -= learning_rate * gradients["W3"]
        self.b3 -= learning_rate * gradients["b3"]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def train(
        self,
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        learning_rate=0.01,
    ):
        loss_history = []
        rng = np.random.default_rng(42)
        n_samples = X_train.shape[0]

        for epoch in range(epochs):

            # Shuffle training data
            indices = rng.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # Process mini-batches
            for start in range(0, n_samples, batch_size):
                end = start + batch_size

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward propagation
                predictions, cache = self.forward(X_batch)

                # Backpropagation
                gradients = self.backward(y_batch, cache)

                # Update weights and biases
                self.update_parameters(
                    gradients,
                    learning_rate,
                )

            # Calculate loss after each epoch
            predictions, _ = self.forward(X_train)
            epoch_loss = cross_entropy_loss(
                y_train,
                predictions,
            )

            loss_history.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Loss: {epoch_loss:.4f}"
                )

        return loss_history

    # --------------------------------------------------------
    # Numerical gradient checking
    # --------------------------------------------------------

    def gradient_check(
        self,
        X,
        y_true,
        num_checks=10,
        epsilon=1e-5,
    ):
        """
        Compare analytical gradients with central finite differences.

        Avoid ReLU nondifferentiable points where perturbations
        change the activation masks.
        """

        _, cache = self.forward(X)
        analytical = self.backward(y_true, cache)

        parameters = {
            "W1": self.W1,
            "W2": self.W2,
            "W3": self.W3,
        }

        rng = np.random.default_rng(123)
        checks = []
        max_attempts = 200
        attempt = 0

        while len(checks) < num_checks and attempt < max_attempts:
            attempt += 1

            name = rng.choice(list(parameters.keys()))
            parameter = parameters[name]

            index = tuple(
                rng.integers(0, size)
                for size in parameter.shape
            )

            analytical_gradient = analytical[name][index]

            # Skip negligible gradients
            if abs(analytical_gradient) < 1e-7:
                continue

            original_value = parameter[index]

            try:
                # Positive perturbation
                parameter[index] = original_value + epsilon
                plus_predictions, cache_plus = self.forward(X)
                loss_plus = cross_entropy_loss(
                    y_true,
                    plus_predictions,
                )

                # Negative perturbation
                parameter[index] = original_value - epsilon
                minus_predictions, cache_minus = self.forward(X)
                loss_minus = cross_entropy_loss(
                    y_true,
                    minus_predictions,
                )

            finally:
                # Always restore the original parameter
                parameter[index] = original_value

            # Skip if a ReLU activation crosses zero
            kink_crossed = (
                np.any(
                    (cache_plus["Z1"] > 0)
                    != (cache_minus["Z1"] > 0)
                )
                or np.any(
                    (cache_plus["Z2"] > 0)
                    != (cache_minus["Z2"] > 0)
                )
            )

            if kink_crossed:
                continue

            # Central finite-difference approximation
            numerical_gradient = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            denominator = max(
                1e-8,
                abs(numerical_gradient)
                + abs(analytical_gradient),
            )

            relative_error = (
                abs(numerical_gradient - analytical_gradient)
                / denominator
            )

            checks.append(
                (
                    name,
                    index,
                    analytical_gradient,
                    numerical_gradient,
                    relative_error,
                )
            )

            print(
                f"{name}{index}: "
                f"analytical={analytical_gradient:.8e}, "
                f"numerical={numerical_gradient:.8e}, "
                f"relative_error={relative_error:.8e}"
            )

        # Ensure the requested number of checks was completed
        if len(checks) < num_checks:
            raise RuntimeError(
                f"Gradient check incomplete: checked "
                f"{len(checks)} of {num_checks} required weights "
                f"after {max_attempts} attempts."
            )

        passed = sum(
            check[4] < 1e-6
            for check in checks
        )

        print(
            f"\nGradient check: {passed}/{len(checks)} passed "
            f"(relative error < 1e-6)"
        )

        # The lab requires at least 9 out of 10 checks to pass
        if passed < 9:
            raise AssertionError(
                f"Gradient check failed: only "
                f"{passed}/{len(checks)} passed."
            )

        return passed, checks


# ============================================================
# 4. BENCHMARK TIMING
# ============================================================

def benchmark_timing(
    model,
    X_batch,
    y_batch,
    num_runs=100,
):
    """
    Measure forward and backward propagation separately.

    Return median execution times in seconds.
    """

    # Warm-up
    _, cache = model.forward(X_batch)
    _ = model.backward(y_batch, cache)

    forward_times = []
    backward_times = []

    for _ in range(num_runs):

        # Forward propagation timing
        t0 = time.perf_counter()
        _, cache = model.forward(X_batch)
        t1 = time.perf_counter()

        forward_times.append(t1 - t0)

        # Backward propagation timing
        t2 = time.perf_counter()
        _ = model.backward(y_batch, cache)
        t3 = time.perf_counter()

        backward_times.append(t3 - t2)

    median_forward = float(np.median(forward_times))
    median_backward = float(np.median(backward_times))

    return median_forward, median_backward


# ============================================================
# 5. EXPERIMENTS AND PLOTTING
# ============================================================

def run_experiments():

    # Store results relative to this Python file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(project_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Training samples: {X_train.shape[0]}, "
        f"Features: {X_train.shape[1]}"
    )
    print(
        f"Testing samples:  {X_test.shape[0]}, "
        f"Features: {X_test.shape[1]}"
    )
    print(f"Output classes:   {y_train.shape[1]}")

    # --------------------------------------------------------
    # 1. Numerical gradient checking
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("1. NUMERICAL GRADIENT CHECKING")
    print("=" * 60)

    check_model = NeuralNetwork(
        hidden_size=16,
        seed=42,
    )

    check_model.gradient_check(
        X_train[:5],
        y_train[:5],
        num_checks=10,
        epsilon=1e-5,
    )

    # --------------------------------------------------------
    # 2. Timing measurements
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("2. TIMING MEASUREMENTS (100 RUNS)")
    print("=" * 60)

    timing_model = NeuralNetwork(
        hidden_size=64,
        seed=42,
    )

    sample_X = X_train[:32]
    sample_y = y_train[:32]

    med_fwd, med_bwd = benchmark_timing(
        timing_model,
        sample_X,
        sample_y,
        num_runs=100,
    )

    print(
        f"Batch size timed: {sample_X.shape[0]} samples, "
        f"hidden layer width: 64"
    )
    print(
        f"Median Forward  Propagation Time: "
        f"{med_fwd * 1e3:.4f} ms "
        f"({med_fwd * 1e6:.1f} µs)"
    )
    print(
        f"Median Backward Propagation Time: "
        f"{med_bwd * 1e3:.4f} ms "
        f"({med_bwd * 1e6:.1f} µs)"
    )

    # --------------------------------------------------------
    # 3. Training experiments
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("3. TRAINING EXPERIMENTS (6 CONFIGURATIONS)")
    print("=" * 60)

    hidden_sizes = [16, 64]
    learning_rates = [0.001, 0.01, 0.1]

    epochs = 50
    batch_size = 32

    results = []
    loss_curves = {}

    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:

            config_name = (
                f"Width {hidden_size}, lr={learning_rate}"
            )

            print(f"\n--- Running: {config_name} ---")

            model = NeuralNetwork(
                hidden_size=hidden_size,
                seed=42,
            )

            losses = model.train(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            # Evaluate on the test set
            test_predictions, _ = model.forward(X_test)

            predicted_classes = np.argmax(
                test_predictions,
                axis=1,
            )
            true_classes = np.argmax(
                y_test,
                axis=1,
            )

            test_accuracy = np.mean(
                predicted_classes == true_classes
            )

            loss_curves[config_name] = losses

            results.append(
                {
                    "hidden_size": hidden_size,
                    "learning_rate": learning_rate,
                    "final_loss": losses[-1],
                    "test_accuracy": test_accuracy,
                }
            )

            print(
                f"Config: {config_name} -> "
                f"Final Loss: {losses[-1]:.4f}, "
                f"Test Acc: {test_accuracy * 100:.2f}%"
            )

    # --------------------------------------------------------
    # Print summary table
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY RESULTS TABLE")
    print("=" * 60)

    print(
        f"{'Hidden Width':<14} | "
        f"{'Learning Rate':<14} | "
        f"{'Final Loss':<12} | "
        f"{'Test Accuracy':<14}"
    )
    print("-" * 60)

    for result in results:
        print(
            f"{result['hidden_size']:<14} | "
            f"{result['learning_rate']:<14} | "
            f"{result['final_loss']:<12.4f} | "
            f"{result['test_accuracy'] * 100:>11.2f}%"
        )

    # --------------------------------------------------------
    # Save experiment results to CSV
    # --------------------------------------------------------

    metrics_path = os.path.join(
        results_dir,
        "experiment_results.csv",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "hidden_size,learning_rate,final_loss,test_accuracy\n"
        )

        for result in results:
            file.write(
                f"{result['hidden_size']},"
                f"{result['learning_rate']},"
                f"{result['final_loss']:.6f},"
                f"{result['test_accuracy']:.6f}\n"
            )

    print(f"\nExperiment results CSV saved to: {metrics_path}")

    # --------------------------------------------------------
    # Save timing results to CSV
    # --------------------------------------------------------

    timing_path = os.path.join(
        results_dir,
        "timing_results.csv",
    )

    with open(
        timing_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "operation,batch_size,hidden_size,num_runs,"
            "median_time_seconds,median_time_ms\n"
        )

        file.write(
            f"forward,32,64,100,"
            f"{med_fwd:.8f},{med_fwd * 1e3:.4f}\n"
        )
        file.write(
            f"backward,32,64,100,"
            f"{med_bwd:.8f},{med_bwd * 1e3:.4f}\n"
        )

    print(f"Timing results CSV saved to: {timing_path}")

    # --------------------------------------------------------
    # Save loss curves plot
    # --------------------------------------------------------

    plot_path = os.path.join(
        results_dir,
        "loss_curves.png",
    )

    plt.figure(figsize=(10, 6))

    colors = {
        0.001: "#1f77b4",
        0.01: "#ff7f0e",
        0.1: "#2ca02c",
    }

    line_styles = {
        16: "--",
        64: "-",
    }

    for config_name, losses in loss_curves.items():

        parts = config_name.split(", ")
        hidden_size = int(parts[0].split()[1])
        learning_rate = float(parts[1].split("=")[1])

        plt.plot(
            range(1, epochs + 1),
            losses,
            label=config_name,
            color=colors[learning_rate],
            linestyle=line_styles[hidden_size],
            linewidth=2,
        )

    plt.title(
        "Training Loss Curves Across Hyperparameter Configurations "
        "(50 Epochs)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(
        title="Configuration",
        fontsize=10,
        title_fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Loss curves saved successfully to: {plot_path}")

    return results, (med_fwd, med_bwd)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_experiments()
