# Individual Project Proposal: EdgeGesture-Lite
## Low-Latency Dynamic Hand Gesture Recognition for Edge Devices via Quantized Temporal CNNs

**Course:** Machine Learning Coursework  
**Student Name:** Muhammadjon Negmatov  
**Student ID:** 68899  
**Institution:** University Machine Learning Laboratory  
**Target Submission Date:** Academic Year 2026/2027  

---

## 1. Problem Statement & Motivation

Touchless human-computer interaction (HCI) is increasingly critical for sterile healthcare consoles, automotive driver-assistance dashboards, smart home controls, and assistive technologies. While standard video-based gesture recognition architectures—such as 3D Convolutional Neural Networks (e.g., I3D, SlowFast) and Video Vision Transformers—achieve high accuracy on cloud infrastructure, their operational profile is fundamentally incompatible with embedded edge systems:
- They incur extreme computational overhead ($\ge 50$ GFLOPs per clip).
- They demand hundreds of megabytes of parameter storage ($\ge 50$ MB).
- They exhibit end-to-end inference latencies exceeding $200\text{--}500\text{ ms}$ on CPU-only edge chips, introducing disorienting interaction lag.

**Objective:** Build **EdgeGesture-Lite**, an end-to-end, ultra-compact dynamic gesture recognition pipeline that operates on continuous streaming 3D hand landmark sequences. By divorcing spatial feature extraction (delegated to optimized upstream landmarkers) from temporal sequence modeling, we can design a specialized 1D Temporal Convolutional Network (TCN) optimized for edge deployment.

---

## 2. Dataset & Preprocessing Pipeline

### 2.1 Dataset Selection
The project will utilize the **IPN Hand (Interaction Pattern Recognition)** and **HaGRID** benchmark datasets, augmented by synthetic multi-subject continuous gesture recordings:
- **Gestures:** 10 core dynamic gestures (Swipe Left, Swipe Right, Swipe Up, Swipe Down, Zoom In/Pinch, Zoom Out/Expand, Rotate CW, Rotate CCW, Open Palm Tap, Fist Confirm).
- **Partition:** Strict subject-independent split: $70\%$ training ($>15{,}000$ clips), $15\%$ validation, and $15\%$ held-out test partitions to ensure generalization to novel hand morphologies.

### 2.2 Mathematical Normalization Pipeline
Raw video frames are converted into temporal windows of $T = 16$ frames at 30 FPS. Each frame contains 21 3D hand keypoints $P_t = [p_{t,0}, p_{t,1}, \dots, p_{t,20}]^T \in \mathbb{R}^{21 \times 3}$. To guarantee absolute invariance to camera distance, subject hand scale, and screen placement, the landmark coordinates undergo a two-stage deterministic transformation:

1. **Wrist-Centric Translation Invariance:**
   $$\tilde{p}_{t,i} = p_{t,i} - p_{t,0} \quad \forall i \in \{0, \dots, 20\}, \, \forall t \in \{1, \dots, T\}$$
2. **Anatomical Scale Normalization:**
   Using the Euclidean distance between the wrist ($p_{t,0}$) and middle finger metacarpophalangeal joint ($p_{t,9}$) as an anatomically invariant scale factor $s_t = \|\tilde{p}_{t,9}\|_2$:
   $$x_{t,i} = \frac{\tilde{p}_{t,i}}{s_t + \epsilon}$$

The flattened vector $v_t \in \mathbb{R}^{63}$ produces an input temporal tensor $X \in \mathbb{R}^{B \times 63 \times 16}$.

---

## 3. Baseline Architecture & Success Metrics

### 3.1 Proposed Model Architecture (Temporal CNN)
- **Input Layer:** 1D Conv over $C_{in} = 63$ channels, kernel size $k=3$, padding $p=1$.
- **Temporal Residual Blocks (x3):** Depthwise separable temporal convolutions (1D Conv $k=3$, BatchNorm, Hardswish, Squeeze-and-Excitation temporal attention, Dropout $0.2$).
- **Classifier Head:** Global Average Pooling (GAP) across the time dimension $\rightarrow$ Linear projection $(128 \rightarrow 10)$.
- **Baseline Complexity:** $\approx 185{,}000$ parameters ($\approx 740\text{ KB}$ in FP32).

### 3.2 Evaluation Metrics
- **Primary Accuracy Metric:** Top-1 Classification Accuracy on the test partition with a target threshold of **$\ge 92.0\%$**.
- **Secondary Quality Metrics:** Macro-averaged F1-Score **$\ge 0.90$** and per-class Confusion Matrix stability across dynamic velocity variations.
- **Robustness Check:** Temporal jitter invariance and zero false-positive activation on background non-gesture hand movement.

---

## 4. Edge Deployment Constraints & Budgets

The system will be deployed and empirically benchmarked on **Apple Silicon (M-series / Neural Engine)** and **Raspberry Pi 4 / ARM Cortex-A72 (1.5 GHz)** single-board computers:

| Constraint Dimension | Mandatory Specification | Project Target (EdgeGesture-Lite) | Safety Margin |
|---|---|---|---|
| **Inference Latency** | **$\le \mathbf{50.0\text{ ms}}$** | **$\le \mathbf{12.5\text{ ms}}$** (Model) / **$\le 35.0\text{ ms}$** (End-to-End) | $4\times$ margin |
| **Model Storage Size** | **$\le \mathbf{5.0\text{ MB}}$** | **$\le \mathbf{0.85\text{ MB}}$** (INT8 uncompressed) | $5.8\times$ margin |
| **Working Memory (RAM)** | $\le 100\text{ MB}$ RSS | $\le 28.5\text{ MB}$ RSS peak working set | $3.5\times$ margin |
| **Power Consumption** | $\le 5.0\text{ W}$ Edge Budget | $\le 1.8\text{ W}$ continuous CPU dissipation | Compliant |

---

## 5. Planned Optimization Strategy

To satisfy and surpass the strict edge deployment criteria, the project executes a two-phase compression and hardware acceleration pipeline:

1. **Post-Training INT8 Quantization (PTQ) & Quantization-Aware Training (QAT):**
   - Quantize 32-bit floating-point weights and activation tensors to signed 8-bit integers (`qint8`):
     $$q = \text{clamp}\left( \left\lfloor \frac{w}{S} \right\rceil + Z, -128, 127 \right)$$
   - Utilize symmetric per-channel scale quantization for convolutional weights and histogram-calibrated asymmetric quantization for activation tensors via PyTorch `torch.ao.quantization` and ONNX Runtime.
   - **Expected Outcome:** $4\times$ reduction in model storage ($740\text{ KB} \rightarrow 185\text{ KB}$) and $2.5\times\text{--}3.2\times$ speedup via ARM NEON SIMD vector instructions.

2. **Knowledge Distillation (Teacher $\rightarrow$ Student):**
   - A heavy Spatio-Temporal Graph Convolutional Network (ST-GCN, $3.2\text{ M}$ parameters) serves as the pre-trained teacher.
   - The lightweight 1D Temporal CNN serves as the student, trained with a combined Loss:
     $$\mathcal{L} = (1 - \alpha) \mathcal{L}_{CE}(y, \sigma(z_s)) + \alpha T^2 \mathcal{L}_{KL}(\sigma(z_s / T), \sigma(z_t / T))$$
   - Recovers any marginal accuracy degradation induced by aggressive model quantization.

3. **Structured Magnitude Pruning:**
   - Pruning $30\%$ of the least significant convolutional filter channels using $\ell_1$-norm ranking, followed by fine-tuning to eliminate dead activations and reduce memory bus cache misses.

---

## 6. Implementation Schedule & Verification Plan

```text
Milestone 1 (Weeks 1-2): Ingestion, landmark normalization pipeline, and Stratified train/val/test splits.
Milestone 2 (Weeks 3-4): FP32 baseline Temporal CNN training, hyperparameter sweep, and MLflow logging.
Milestone 3 (Weeks 5-6): Teacher ST-GCN implementation and Knowledge Distillation training loop.
Milestone 4 (Weeks 7-8): INT8 PTQ/QAT quantization, ONNX Runtime conversion, and latency/memory benchmarking.
Milestone 5 (Weeks 9-10): Final evaluation report, confusion matrices, and interactive edge demo.
```

---

**Student Confirmation:**  
I, **Muhammadjon Negmatov** (Student ID: **68899**), confirm that this proposal represents my original independent work, drawing upon my background in computer vision and landmark preprocessing while formulating a distinct, novel optimization project for edge deployment.
