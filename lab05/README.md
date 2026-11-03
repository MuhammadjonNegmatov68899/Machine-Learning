# Machine Learning Lab 05: Frameworks and Automatic Differentiation

This directory contains the complete, reproducible implementation for **Lab 5: Frameworks and Automatic Differentiation** in Python.

---

## 1. Overview & Objectives

Lab 5 investigates the theoretical foundations, implementation mechanics, and performance trade-offs of modern automatic differentiation and deep learning frameworks:

1. **Scalar Reverse-Mode Automatic Differentiation Engine (`src/engine.py`):**
   - Implements a scalar `Value` node that constructs a dynamic directed acyclic computation graph (DAG) during the forward pass.
   - Computes exact analytical gradients via reverse topological sorting and reverse-mode accumulation (`+=`).
   - Supports fundamental arithmetic (`+`, `*`, `**`, `-`, `/`), non-linear activations (`ReLU`, `Sigmoid`, `Tanh`), transcendental functions (`exp`, `log`), and scalar reductions (`sum`).
2. **Mathematical Gradient Verification (`src/operations.py`):**
   - Evaluates the mandatory assignment expression:
     $$\mathcal{L} = \left((a \cdot b + c)^2\right).\text{relu}().\text{sum}()$$
   - Rigorously checks custom engine gradients against both **PyTorch autograd** and **central finite-difference numerical approximations** ($[f(x+\epsilon) - f(x-\epsilon)] / (2\epsilon)$).
   - Enforces strict tolerance: $\max |\nabla_{\text{custom}} - \nabla_{\text{ref}}| < 1.0 \times 10^{-6}$.
3. **End-to-End Neural Network Training on `make_moons` (`src/training.py`):**
   - Architecture: $2 \rightarrow 8 \rightarrow 1$ Multi-Layer Perceptron (33 trainable parameters).
   - Hidden Activation: Sigmoid; Output Activation: Sigmoid; Objective: Binary Cross-Entropy (BCE).
   - Genuinely trains using the custom AD engine and compares against an equivalent PyTorch model.
   - Measures training duration, convergence trajectories, and reports median metrics across 5 reproducible random seeds (`42, 1337, 2024, 7, 999`).
4. **PyTorch Framework Performance & Memory Layouts (`src/benchmarks.py`):**
   - Inspects backward computational DAGs via `grad_fn` walks and TorchScript Intermediate Representation (IR).
   - Benchmarks inference execution across **Eager**, **Traced** (`torch.jit.trace`), **Scripted** (`torch.jit.script`), and **Compiled** (`torch.compile`) modes.
   - Investigates tensor strides, contiguity (`is_contiguous()`), and hardware cache locality impact on contiguous vs non-contiguous layouts.

---

## 2. Directory Structure

```text
lab05/
├── README.md                          # Setup, execution guide, and empirical findings
├── report.md                          # Comprehensive academic report and analysis
├── requirements.txt                   # Pinned Python package dependencies
├── config.yaml                        # Centralized parameters, seeds, and paths
├── run_experiments.py                 # Single-command executable entry point
├── src/
│   ├── __init__.py                    # Package initialization and exports
│   ├── engine.py                      # Value node & reverse-mode AD engine (DFS topological sort)
│   ├── operations.py                  # Functional primitives, BCE loss, and finite difference verification
│   ├── neural_network.py              # Module, Neuron, Layer, MLP, and PyTorchMLP classes
│   ├── training.py                    # make_moons data prep and 5-seed comparative training
│   └── benchmarks.py                  # Graph inspection, eager vs compiled timing, tensor contiguity
├── tests/
│   ├── conftest.py                    # Shared test fixtures and initialized networks
│   ├── test_engine.py                 # 17 unit tests for Value node, ops, DAG, and accumulation
│   ├── test_gradients.py              # 8 tests for gradient verification (< 1e-6 tolerance)
│   └── test_training.py               # 7 tests for architecture, parity, and loss descent
└── results/                           # Generated experimental artifacts
    ├── gradient_comparison.csv        # Exact gradients and absolute errors vs PyTorch & numerical
    ├── training_comparison.csv        # 5-run metrics and median summaries
    ├── execution_benchmarks.csv       # Eager vs Traced vs Compiled inference latencies
    ├── tensor_layout_benchmarks.csv   # Contiguous vs non-contiguous reduction benchmarks
    ├── benchmark_plots.png            # Publication-ready 4-panel visual summary
    └── summary.md                     # Detailed Markdown execution report
```

---

## 3. Quick Start & Execution

### Prerequisites

Activate the project virtual environment from the repository root:

```bash
source .venv/bin/activate
```

Ensure dependencies are installed:

```bash
pip install -r lab05/requirements.txt
```

### Running All Experiments in a Single Command

Execute the complete end-to-end experimental pipeline:

```bash
python lab05/run_experiments.py
```

This single command automatically:
1. Verifies scalar and vector gradients against PyTorch and numerical finite differences.
2. Trains both Custom AD and PyTorch neural networks across all 5 random seeds on `make_moons`.
3. Inspects the PyTorch backward computation graph and TorchScript IR.
4. Benchmarks Eager vs Traced vs Compiled execution latency and throughput.
5. Benchmarks contiguous vs non-contiguous tensor operations.
6. Generates all CSV tables, markdown summaries, and the multi-panel plot in `lab05/results/`.

---

## 4. Running Automated Tests

Run the full automated test suite (32 unit and integration tests):

```bash
pytest lab05/tests/ -v
```

### Test Suite Coverage:
- `test_engine.py` (17 tests):
  - Initialization, forward and backward passes for `+`, `*`, `**`, `ReLU`, `-`, `/`, `exp`, `log`, `sigmoid`, `tanh`, `sum`.
  - Zero division error handling.
  - Multi-path gradient accumulation: $y = x + x \implies y' = 2$, $y = x \cdot x \implies y' = 2x$.
  - Diamond DAG topologies: $z = (x + y)(x - y) = x^2 - y^2$.
  - Deep multi-branch DAG parity vs PyTorch autograd.
  - Graph-wide gradient resetting (`zero_grad()`).
- `test_gradients.py` (8 tests):
  - Target expression `((a * b + c) ** 2).relu().sum()` across multiple parameter regions.
  - Strict assertions: $\max | \nabla_{\text{custom}} - \nabla_{\text{torch}} | < 1e-6$ and $\max | \nabla_{\text{custom}} - \nabla_{\text{num}} | < 1e-6$.
  - Analytical calculus parity check ($-24.0, 16.0, 8.0$).
  - 5-dimensional vector gradient verification.
- `test_training.py` (7 tests):
  - Parameter counts for Neuron, Layer, MLP, and PyTorchMLP ($2 \rightarrow 8 \rightarrow 1 = 33$ parameters).
  - Forward output parity with synchronized weights ($|y_{\text{custom}} - y_{\text{torch}}| < 1e-5$).
  - Backward gradient parity with synchronized weights ($|\nabla W_{\text{custom}} - \nabla W_{\text{torch}}| < 1e-4$).
  - Training loss reduction on `make_moons`.
  - Deterministic reproducibility with fixed seeds.

---

## 5. Summary of Empirical Findings (Synchronized with Results CSVs)

### Task 4: Gradient Verification
For the target expression `((a * b + c) ** 2).relu().sum()` with $a = 2.0, b = -3.0, c = 10.0$:
- **Analytical Values:** $\nabla_a = -24.0$, $\nabla_b = 16.0$, $\nabla_c = 8.0$.
- **Error vs PyTorch Autograd:** $0.000 \times 10^0$ (exact bit-level match).
- **Error vs Central Finite Differences ($\epsilon = 10^{-6}$):**
  - Variable $a$: $1.086 \times 10^{-9}$
  - Variable $b$: $2.236 \times 10^{-9}$
  - Variable $c$: $6.875 \times 10^{-9}$
- **Verdict:** **PASSED** (all errors $< 1.0 \times 10^{-6}$ by more than 2 orders of magnitude).

### Task 5: `make_moons` Training (Median of 5 Runs)
- **Model Architecture:** $2 \rightarrow 8 \rightarrow 1$ (Sigmoid hidden + Sigmoid output, 33 parameters).
- **Optimization:** Full-batch gradient descent (150 samples), $\eta = 0.5$, 150 epochs.
- **Weight Initialization Note:** Both models are initialized using the same Xavier uniform parameter distribution formula ($\mathcal{U}(-\sqrt{6/(n_{in}+1)}, \sqrt{6/(n_{in}+1)})$) with zero bias across the 5 seeds, but via their respective native random number generators (`numpy.random.default_rng` for the Custom Engine and `torch.manual_seed` for PyTorch). They therefore draw distinct random realizations from the same distribution, rather than identical numerical weights. (Exact weight synchronization is verified separately in unit tests).
- **Results:**
  - **Custom Engine:** Train Loss: `0.3426`, Test Loss: `0.2663`, Train Accuracy: `81.33%`, Test Accuracy: `90.00%`, Training Time: `9.486 s`.
  - **PyTorch Engine:** Train Loss: `0.3450`, Test Loss: `0.2688`, Train Accuracy: `81.33%`, Test Accuracy: `90.00%`, Training Time: `0.018 s`.
- **Finding:** Both engines converge to statistically indistinguishable decision boundaries and loss values. PyTorch is approximately $531.1\times$ faster due to vectorized C++ ATen kernels compared to pure Python scalar object pointer dereferencing.

### Task 6: Framework Execution & Memory Layout
- **Eager vs Traced vs Compiled (Batch Size = 32, 1,000 Timed Iterations):**
  - **Eager:** Median latency = `9.96 µs` (Throughput: 3,159,222 inf/s).
  - **Traced (`torch.jit.trace`):** Median latency = `7.13 µs` (Throughput: 4,357,748 inf/s), **`1.40x` speedup**.
  - **Scripted (`torch.jit.script`):** Median latency = `6.12 µs` (Throughput: 5,172,434 inf/s), **`1.63x` speedup**.
  - **Compiled (`aot_eager`):** Median latency = `77.46 µs` ($0.13\times$ speedup, Dynamo frame guard overhead dominates for tiny micro-architectures).
  - **Compiled (`inductor`):** UNSUPPORTED on this local environment (raises `BackendCompilerFailed: backend='inductor' raised: CppCompileError: C++ compile error`).
- **Memory Contiguity (Row-Wise Reduction on $2048 \times 2048$ Matrix):**
  - **Contiguous (Row-Major, stride = `[2048, 1]`):** `0.182 ms` ($92.37\text{ GB/s}$ throughput).
  - **Non-Contiguous (Transposed View, stride = `[1, 2048]`):** `0.495 ms` ($33.89\text{ GB/s}$ throughput).
  - **Slowdown Factor:** **$2.73\times$ slower** on non-contiguous layout due to CPU cache line misses and strided RAM access.
