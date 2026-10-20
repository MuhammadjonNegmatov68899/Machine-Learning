# Machine Learning Coursework

This repository brings together the complete laboratory assignments and coursework for the **Machine Learning** university course. Each laboratory is maintained as an isolated, self-contained project with its own source code, dependencies, empirical measurements, result tables, plots, and technical report.

---

## Repository Structure

```text
Machine-Learning/
├── README.md                      # Repository overview and lab navigation
├── requirements.txt               # Shared reference dependencies
├── .gitignore                     # Repository-wide ignore rules (.venv, data, caches)
│
├── lab01/                         # Lab 1: Profiling and Multi-Device Deployment
│   ├── report.md                  # Comprehensive Lab 1 technical report
│   ├── results/                   # Accuracy, measurements, and TinyML resource budgets
│   └── src/                       # Baseline models, timing scripts, and version tracking
│
├── lab02/                         # Lab 2: Multi-Layer Neural Network from Scratch
│   ├── README.md                  # Lab 2 setup and reproduction instructions
│   ├── report.md                  # Lab 2 technical report (NumPy MLP, backprop, finite diffs)
│   ├── requirements.txt           # Lab 2 dependencies
│   ├── results/                   # Experiment results, timing, and loss curves
│   └── src/                       # Custom NumPy multi-layer perceptron implementation
│
└── lab03/                         # Lab 3: Architectures and Their Costs
    ├── README.md                  # Lab 3 execution guide and command reference
    ├── report.md                  # Lab 3 technical report (MLP vs CNN vs Vision Transformer)
    ├── requirements.txt           # Lab 3 dependencies (PyTorch, TorchInfo, THOP)
    ├── run_experiments.py         # Main execution driver for training, profiling, and benchmarking
    ├── results/                   # Trade-off plots and complete CSV metrics
    └── src/                       # Dataset loading, models, FLOPs/MACs cost profiling, latency
```

---

## Laboratories Overview

### [Lab 1: Multi-Device Deployment & Profiling](lab01/)
- **Focus**: Edge and TinyML constraints, model profiling, and hardware resource budgeting.
- **Key Deliverables**: Logistic Regression and Random Forest models on tabular data, memory footprint and inference latency profiling, and micro-controller deployment feasibility analysis.
- **Documentation**: [Lab 1 Technical Report](lab01/report.md)

### [Lab 2: Multi-Layer Neural Network from Scratch](lab02/)
- **Focus**: First-principles implementation of a Multi-Layer Perceptron (MLP) using only pure NumPy.
- **Key Deliverables**: Manual implementation of forward propagation, analytical backpropagation via the chain rule, two-sided numerical gradient checking, mini-batch gradient descent, and hyperparameter sensitivity sweeps across hidden layer dimensions and learning rates.
- **Documentation**: [Lab 2 README](lab02/README.md) • [Lab 2 Technical Report](lab02/report.md)

### [Lab 3: Architectures and Their Costs](lab03/)
- **Focus**: Comparative study of three fundamental neural network paradigms—**MLP**, **CNN**, and **Vision Transformer (Attention)**—on Fashion-MNIST.
- **Key Deliverables**:
  - Parameter derivations and automated counting via `torchinfo.summary`.
  - Theoretical FLOPs vs. MACs ($1\text{ MAC} = 2\text{ FLOPs}$) using Reddi Chapter 4 formulas, contrasted against `thop.profile` with custom hooks for native Transformer layers.
  - Repeatable inference latency benchmarking across batch sizes 1 (100 runs) and 128 (20 runs) on Apple Silicon MPS and CPU with warmup and hardware synchronization.
  - 5-epoch training dynamics with Adam ($\text{lr}=10^{-3}$, batch size $128$).
  - Architectural and system-level trade-offs (operational intensity, sliding-window cache reuse vs. memory streaming).
- **Documentation**: [Lab 3 README](lab03/README.md) • [Lab 3 Technical Report](lab03/report.md)

---

## Environment & Setup

Each laboratory contains its own pinned `requirements.txt`. To set up a global environment:

```bash
# Clone the repository
git clone https://github.com/MuhammadjonNegmatov68899/Machine-Learning.git
cd Machine-Learning

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (e.g., for Lab 3)
pip install -r lab03/requirements.txt
```
