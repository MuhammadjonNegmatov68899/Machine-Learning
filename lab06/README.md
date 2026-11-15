# Machine Learning Lab 06: Training Pipelines, Hardware Profiling, and Project Proposal

**Author:** Muhammadjon Negmatov  
**Student ID:** 68899  
**Institution:** University Machine Learning Laboratory  
**Environment:** macOS 27.2 (Darwin arm64), Apple Silicon M2 (8 GB Unified Memory), Python 3.11.8, PyTorch 2.3.1, MLflow 2.14.1  

---

## 1. Overview & Objectives

Lab 6 builds an end-to-end, reproducible, production-grade deep learning engineering pipeline around the best-performing neural architecture from Lab 3 (**Convolutional Neural Network - CNN** on Fashion-MNIST). The laboratory executes 8 rigorous technical deliverables spanning MLOps, system profiling, data loading concurrency, hardware-constrained batch sizing, optimizer/scheduler dynamics, gradient accumulation, stateful checkpoint resumption, and an individual project proposal tailored to edge deployment.

### Key Laboratory Deliverables

1. **Task 1: YAML-Driven Training Pipeline (20 pts):**
   - Single YAML configuration object (`lab06/configs/train_config.yaml`) defining random seeds, dataset parameters, CNN model dimensions, training hyperparameters, checkpoint intervals, and MLflow tracking.
   - Deterministic seed orchestration across Python `random`, NumPy, and PyTorch (including DataLoader generators).
   - Stateful checkpoint management saving weights, optimizer states, scheduler states, metrics, and complete RNG states.
   - MLflow integration logging flattened parameters, epoch-level metrics (`train_loss`, `val_loss`, `val_acc`, `throughput`), final test metrics, and artifacts (best model checkpoint, config YAML, loss curves).

2. **Task 2: PyTorch Profiling & Compute Breakdown (15 pts):**
   - Micro-architectural execution profiling over a complete training epoch via `torch.profiler`.
   - Granular decomposition and percentage share reporting for:
     - **Data Loading**
     - **Forward Pass**
     - **Backward Pass**
     - **Optimizer Step**
   - Export of Kineto Chrome traces (`lab06/results/task2_trace.json`), tabular operator summaries, and publication-ready breakdown visualizations.

3. **Task 3: DataLoader Workers Concurrency Study (15 pts):**
   - Parametric empirical sweep across worker subprocesses: $\text{num\_workers} \in [0, 1, 2, 4, 8]$.
   - Measurement of epoch latency and sample throughput (samples/sec).
   - In-depth hardware bottleneck identification on Apple Silicon unified memory (inter-process communication / IPC context switching vs. background prefetching).

4. **Task 4: Batch Size & Hardware Trade-Off Study (15 pts):**
   - Empirical sweep across batch sizes: $\text{batch\_size} \in [32, 64, 128, 256, 512]$.
   - Dual measurement of compute throughput (samples/sec) and peak process RSS memory (MB).
   - Evaluation of generalization performance (test accuracy) under an exact 5-epoch training budget.

5. **Task 5: Optimizers and Learning Rate Schedulers (15 pts):**
   - Full factorial comparison of 2 optimizers (**SGD with momentum** $\mu=0.9$, **Adam**) across 3 schedulers (**StepLR**, **CosineAnnealingLR**, **ReduceLROnPlateau**) under equal 10-epoch budgets (6 total configurations).
   - Comparative analysis of convergence trajectories, training loss, validation loss, final test accuracy, and execution latency.

6. **Task 6: Gradient Accumulation (10 pts):**
   - Implementation of micro-batch gradient accumulation to simulate target batch size $256$ using physical batch size $64$ with $K=4$ accumulation steps.
   - Rigorous mathematical and empirical comparison against standard physical batch size $256$.
   - Demonstration of significant working memory savings with virtually identical convergence curves.

7. **Task 7: Stateful Checkpoint Interruption and Resumption (10 pts):**
   - Verification of uninterrupted training vs. an interrupted run halted at Epoch 3 and resumed from disk.
   - Comprehensive restoration of model weights, optimizer momentum buffers, learning rate scheduler state, and PRNG states.
   - Verification of continuous, seamless loss and accuracy trajectories.

8. **Task 8: Individual Project Proposal (15 pts):**
   - A one-page individual project proposal (**EdgeGesture-Lite**) authored by **Muhammadjon Negmatov (Student ID: 68899)**.
   - Focuses on real-time 3D hand landmark sequence classification on edge CPUs under strict physical constraints: **Latency $\le 50\text{ ms}$** and **Model Size $\le 5\text{ MB}$**, utilizing INT8 quantization and structured pruning.

---

## 2. Directory Structure

```text
lab06/
├── configs/
│   └── train_config.yaml              # Single master YAML configuration driving the pipeline
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── models.py                      # CNN architecture (20,490 params) from Lab 3
│   ├── dataset.py                     # Fashion-MNIST loader, splits (55k/5k/10k), deterministic seeds
│   ├── pipeline.py                    # YAML pipeline driver, checkpointing, MLflow logging
│   ├── profiler.py                    # Task 2: PyTorch profiler breakdown & Chrome trace export
│   ├── studies.py                     # Tasks 3-7: Workers, Batch Size, Optimizers, Grad Accum, Resume
│   └── utils.py                       # Seeding, memory tracking, device selection, plot styles
├── scripts/
│   ├── run_pipeline.py                # Standalone CLI runner for YAML-driven pipeline (Task 1)
│   └── run_all_experiments.py         # Master orchestrator executing Tasks 1 through 7
├── results/                           # Generated CSV study tables, profiler summaries, and traces
│   ├── task2_profiling_breakdown.csv
│   ├── task2_profiler_summary.txt
│   ├── task2_trace.json
│   ├── task3_workers_study.csv
│   ├── task4_batch_size_study.csv
│   ├── task5_optimizers_schedulers_study.csv
│   ├── task5_training_histories.csv
│   ├── task6_gradient_accumulation.csv
│   └── task7_checkpoint_resume.csv
├── plots/                             # Generated high-resolution publication plots (300 DPI)
│   ├── training_curves.png
│   ├── task2_profiling_breakdown.png
│   ├── task3_workers_study.png
│   ├── task4_batch_size_tradeoffs.png
│   ├── task5_loss_curves.png
│   ├── task5_val_accuracy_curves.png
│   ├── task5_final_metrics_bar.png
│   ├── task6_grad_accum_comparison.png
│   └── task7_resume_curves.png
├── checkpoints/                       # Saved model checkpoints (.pt)
│   ├── best_model.pt
│   └── interrupted_epoch_3.pt
├── reports/
│   └── report.md                      # Comprehensive academic report for Lab 6
├── proposal/
│   └── project_proposal.md            # Task 8: Individual project proposal (Muhammadjon Negmatov, ID 68899)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures and path configuration
│   ├── test_models.py                 # Architecture dimensions & parameter counting tests
│   ├── test_dataset.py                # Split shapes & deterministic loader tests
│   ├── test_checkpoint.py             # Checkpoint serialization & state restoration tests
│   ├── test_grad_accum.py             # Gradient accumulation numerical equivalence tests
│   └── test_pipeline.py               # End-to-end YAML pipeline integration smoke test
├── requirements.txt                   # Lab 6 pinned dependencies
└── README.md                          # Master documentation (this file)
```

---

## 3. Installation & Setup

### 3.1 Virtual Environment
Ensure Python 3.11+ is available. From the repository root:

```bash
# Activate existing virtual environment
source .venv/bin/activate

# Install required dependencies
pip install -r lab06/requirements.txt
```

### 3.2 Dataset Preparation
The pipeline automatically resolves the Fashion-MNIST dataset from `lab03/data` if present, avoiding redundant downloads. If running in a fresh workspace, the dataset will download automatically into `lab06/data`.

---

## 4. Execution Guide

### 4.1 Running Automated Unit Tests
To verify all model components, checkpointing mechanics, gradient accumulation equivalence, and deterministic seeds:

```bash
pytest lab06/tests -v
```

### 4.2 Task 1: Running the YAML-Driven Training Pipeline
To run the standard reproducible pipeline driven by `lab06/configs/train_config.yaml`:

```bash
python lab06/scripts/run_pipeline.py --config lab06/configs/train_config.yaml --device auto
```

To launch the MLflow Tracking UI and inspect logged parameters, metrics, and artifacts:

```bash
mlflow ui --backend-store-uri lab06/mlruns
```

### 4.3 Running All Laboratory Experiments (Tasks 1 through 7)
To execute the complete empirical suite sequentially:

```bash
python lab06/scripts/run_all_experiments.py --tasks all --device auto
```

Or execute specific task subsets (e.g. Profiling and Workers study):

```bash
python lab06/scripts/run_all_experiments.py --tasks 2,3 --device auto
```

---

## 5. Hardware Optimization on Apple Silicon M2 (8 GB RAM)

This codebase is specifically engineered for resource-constrained macOS Apple Silicon hardware:
1. **Device Selection:** Automatically detects Apple Silicon Metal Performance Shaders (`mps`) when available, falling back gracefully to CPU.
2. **Kineto Profiler Schedule:** Uses a targeted warm-up and active step schedule (`wait=2, warmup=3, active=25`) for `torch.profiler` with `record_shapes=False` to prevent memory blowup and pybind11 hash collisions.
3. **Memory Monitoring:** Employs `psutil` to track real resident set size (RSS) memory alongside device memory allocations.
4. **Multiprocessing IPC Awareness:** Accounts for Python multiprocessing fork/spawn behavior on macOS to highlight why large worker counts degrade performance on small in-memory datasets.

---

## 6. Student Project Proposal Summary

The full one-page individual project proposal is archived at:  
👉 [`lab06/proposal/project_proposal.md`](proposal/project_proposal.md)

- **Student:** Muhammadjon Negmatov (ID: 68899)
- **Title:** EdgeGesture-Lite: Low-Latency Real-Time Hand Gesture Recognition on Edge CPUs via Quantized Temporal CNNs
- **Key Specifications:**
  - **Latency:** $\le 12.5\text{ ms}$ classifier latency ($\le 35\text{ ms}$ end-to-end), well below the $\le 50\text{ ms}$ ceiling.
  - **Model Size:** $\approx 0.85\text{ MB}$ uncompressed INT8 weights, well below the $\le 5\text{ MB}$ limit.
  - **Planned Optimizations:** Post-Training INT8 Quantization (PTQ), Knowledge Distillation from a Spatio-Temporal Graph Convolutional Network (ST-GCN), and Structured Channel Pruning.
