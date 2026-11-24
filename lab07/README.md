# Lab 7: Compression — Pruning and Distillation

This directory contains the complete implementation, evaluation suite, and academic report for **Lab 7: Compression: Pruning and Distillation**, adhering to Chapter 9 (*Efficient AI*) and Chapter 10 (*Model Optimizations*) of Reddi et al.

---

## 1. Directory Structure

```text
lab07/
├── configs/
│   └── config.yaml             # Experiment hyperparameters & hardware configuration
├── src/
│   ├── __init__.py
│   ├── models.py               # BaselineCNN (~1M params) & StudentCNN (~100k params)
│   ├── dataset.py              # Fashion-MNIST dataloader pipeline
│   ├── training.py             # Standard training & 3-epoch fixed-mask fine-tuning
│   ├── pruning.py              # Global, layer-wise, and structured filter pruning
│   ├── distillation.py         # Hinton Knowledge Distillation loss & training loop
│   ├── benchmark.py            # Latency (CPU & MPS) and dense vs sparse storage analysis
│   ├── visualization.py        # Publication-grade plotting routines
│   ├── efficiency.py           # Chapter 9 Four Pillars synthesis (Data, Model, Compute, Energy)
│   └── utils.py                # Reproducibility, parameter counting, and FLOP calculation
├── scripts/
│   ├── train_baseline.py       # Task 1: Train oversized baseline CNN
│   ├── run_pruning_experiments.py # Task 2 & 3: Pruning and storage/latency benchmarks
│   ├── run_distillation_experiments.py # Task 4: Plain vs distilled student training
│   ├── run_efficiency_analysis.py # Task 5: Four-pillar efficiency aggregation
│   └── run_all_experiments.py  # Master orchestrator running the entire pipeline
├── tests/
│   ├── conftest.py             # Shared fixtures and path resolution
│   ├── test_models.py          # Model architecture & parameter counts (~1M and ~100k)
│   ├── test_pruning.py         # Global & layer-wise sparsity verification
│   ├── test_mask_retention.py  # Fixed-mask retention during optimizer updates
│   ├── test_structured_pruning.py # Physical channel reduction & dimension consistency
│   ├── test_distillation.py    # Distillation loss, numerical stability, teacher freezing
│   ├── test_checkpoint.py      # Checkpoint serialization roundtrip
│   └── test_smoke.py           # End-to-end smoke tests
├── results/                    # CSV metrics, JSON profiles, and Markdown summaries
├── plots/                      # High-resolution (300 DPI) comparison curves
├── checkpoints/                # Saved model checkpoints (.pt)
├── reports/
│   └── report.md               # Comprehensive academic report
├── requirements.txt
└── README.md
```

---

## 2. Environment Setup

The codebase runs under Python 3.11 with PyTorch 2.3+ on Apple Silicon (MPS) or standard CPU/CUDA environments:

```bash
# Activate environment
source ../.venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 3. Running the Automated Test Suite

To verify all 35 unit, integration, and smoke tests:

```bash
pytest lab07/tests/ -v
```

---

## 4. Reproducing the Experiments

You can execute the entire pipeline with a single master command:

```bash
PYTHONPATH=. python3 lab07/scripts/run_all_experiments.py
```

Or execute individual tasks step by step:

### Task 1: Oversized Baseline CNN Training (~1M parameters)
```bash
PYTHONPATH=. python3 lab07/scripts/train_baseline.py
```
- Trains `BaselineCNN` on Fashion-MNIST.
- Saves checkpoint to `lab07/checkpoints/baseline_teacher.pt`.
- Profiles inference latency, FLOPs/MACs, and disk size.

### Task 2 & 3: Magnitude Pruning, Structured Pruning, and Latency Analysis
```bash
PYTHONPATH=. python3 lab07/scripts/run_pruning_experiments.py
```
- Evaluates Global Magnitude Pruning: `[0, 0.25, 0.50, 0.75, 0.90, 0.95]`.
- Evaluates Layer-wise Magnitude Pruning: `[0, 0.25, 0.50, 0.75, 0.90, 0.95]`.
- Evaluates Structured Filter Pruning: `[10%, 25%, 50%]`.
- Fine-tunes each pruned variant for 3 epochs with fixed masks.
- Measures dense vs sparse on-disk storage and CPU/MPS inference latency.
- Generates publication-ready figures in `lab07/plots/`.

### Task 4: Knowledge Distillation
```bash
PYTHONPATH=. python3 lab07/scripts/run_distillation_experiments.py
```
- Trains plain `StudentCNN` (~100k parameters) with Cross-Entropy only.
- Trains distilled `StudentCNN` with Hinton Knowledge Distillation ($T=4, \alpha=0.5$) using the frozen baseline teacher.
- Compares generalization, convergence trajectories, and accuracy.

### Task 5: Four Pillars Efficiency Analysis
```bash
PYTHONPATH=. python3 lab07/scripts/run_efficiency_analysis.py
```
- Compiles synthesis metrics across Data, Model, Compute, and Energy pillars.
- Outputs `lab07/results/four_pillars_efficiency.csv` and `lab07/results/four_pillars_table.md`.

---

## 5. Cloud GPU Execution via Kaggle Notebooks

To execute the remaining tasks on cloud GPUs without local hardware constraints:
1. Upload `lab07.zip` as a Kaggle Dataset.
2. Import `lab07/kaggle/lab07_kaggle.ipynb` into a new Kaggle Notebook.
3. Attach the dataset, enable **GPU T4 x1**, and run cells sequentially.
4. Download the generated `lab07_completed_results.zip` directly from the Kaggle Output pane.
