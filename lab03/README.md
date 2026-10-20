# Machine Learning Lab 3: Architectures and Their Costs

This repository contains the complete, reproducible implementation for **Lab 3: Architectures and Their Costs**. It evaluates three fundamental neural network paradigms—**Multi-Layer Perceptron (MLP)**, **Convolutional Neural Network (CNN)**, and **Vision Transformer (Attention)**—on the Fashion-MNIST benchmark. The project measures compute cost (parameters, MACs, FLOPs), inference latency across batch sizes 1 and 128, and empirical classification accuracy across 5 training epochs.

---

## 1. Project Directory Structure

```text
lab03/
├── README.md                      # Setup and execution guide
├── report.md                      # Comprehensive technical report and theoretical analysis
├── requirements.txt               # Exact dependencies and library versions
├── run_experiments.py             # Main entry point to run profiling, latency, training, and plotting
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Dataset loading, normalization, and 55k/5k train/val split
│   ├── models.py                  # PyTorch implementations of MLP, CNN, and Attention models
│   ├── compute_cost.py            # Hand calculations, torchinfo summary, and thop profiling
│   ├── latency.py                 # Synchronized inference latency benchmarking (batch 1 and 128)
│   ├── train.py                   # 5-epoch training loop with Adam (lr=1e-3, batch=128)
│   └── plot_results.py            # Generation of publication-quality trade-off plots
└── results/                       # Saved CSVs and visualization figures
    ├── summary_metrics.csv        # Overall model metrics (params, FLOPs, latency, accuracy)
    ├── latency_results.csv        # Detailed latency statistics (mean, median, std)
    ├── profiler_comparison.csv    # Comparison of hand calculations, torchinfo, and thop
    ├── layer_hand_costs.csv       # Step-by-step layer FLOPs/MACs derivations
    ├── training_history.csv       # Epoch-by-epoch loss and accuracy metrics
    ├── accuracy_vs_params.png     # Accuracy vs. parameter count plot
    ├── accuracy_vs_latency.png    # Accuracy vs. inference latency plot (Batch 1 and 128)
    └── training_curves.png        # Training loss and validation accuracy curves
```

---

## 2. Environment Setup

It is recommended to run inside a virtual environment with Python 3.10 or 3.11:

```bash
# Navigate to the lab03 directory
cd lab03

# Create and activate a virtual environment (if not already activated)
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 3. Running the Experiments

The experiment driver `run_experiments.py` supports multiple execution modes:

### A. Run All Architectures (Recommended)
Runs both the exact manual specifications and the calibrated ~100k–150k parameter models:
```bash
python run_experiments.py --arch-mode all
```

### B. Run Exact Manual Specifications Only
Runs the exact layer dimensions stated in the manual bullet points:
```bash
python run_experiments.py --arch-mode specified
```

### C. Run Calibrated ~100k–150k Parameter Models Only
Runs the models calibrated to land directly inside the 100,000–150,000 parameter target window:
```bash
python run_experiments.py --arch-mode calibrated
```

### Optional Command-Line Flags
- `--device {auto,cpu,mps,cuda}`: Force execution device (default: `auto`, automatically selects Apple Silicon MPS or CUDA if available, falling back to CPU).
- `--epochs <N>`: Training epochs (default: `5`).
- `--batch-size <N>`: Mini-batch size (default: `128`).
- `--lr <val>`: Adam learning rate (default: `1e-3`).
- `--seed <N>`: Random seed for reproducible dataset splitting and model weight initialization (default: `42`).

---

## 4. Key Results Summary

All results are automatically evaluated and written to `results/`:

| Architecture | Mode | Parameters | Hand FLOPs | thop MACs | Batch 1 Latency | Batch 128 Latency | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | Specified | 235,146 | 469,504 | 234,752 | ~0.027 ms (CPU) / 0.36 ms (MPS) | ~0.093 ms (CPU) / 0.45 ms (MPS) | ~87.4% |
| **CNN** | Specified | 20,490 | 2,063,488 | 1,031,744 | ~0.095 ms (CPU) / 0.44 ms (MPS) | ~4.93 ms (CPU) / 1.24 ms (MPS) | ~89.5% |
| **Attention** | Specified | 71,818 | 2,329,856 | 50,816* | ~0.261 ms (CPU) / 1.77 ms (MPS) | ~6.05 ms (CPU) / 3.19 ms (MPS) | ~82.9% |

*\*Note on thop*: `thop` does not support PyTorch's native `nn.TransformerEncoderLayer`, reporting only the patch projection and linear head operations. Full mathematical derivations and explanations are detailed in `report.md`.
