# Individual Project Proposal: EdgeGesture-Lite
## Low-Latency Dynamic Hand Gesture Recognition for Edge Devices via Quantized Temporal CNNs

**Student Name:** Muhammadjon Negmatov | **Student ID:** 68899 | **Target:** Edge Embedded Deployment

---

### 1. Problem Statement & Motivation
Touchless human-computer interaction (HCI) is essential for sterile surgical consoles, automotive cockpits, and assistive technologies. Standard video 3D CNNs (e.g., I3D, SlowFast) require $>50\text{ GFLOPs}$ and $>50\text{ MB}$ of storage, causing latency $>200\text{ ms}$ on edge CPUs. **EdgeGesture-Lite** decouples spatial landmark extraction from temporal modeling, using a compact 1D Temporal Convolutional Network (TCN) on streaming 3D hand keypoints to achieve real-time, low-power edge inference.

---

### 2. Dataset & Mathematical Preprocessing
- **Dataset:** IPN Hand and HaGRID benchmark datasets (10 dynamic gestures: Swipe, Zoom/Pinch, Rotate, Tap, Fist), partitioned into subject-independent 70% train ($>15{,}000$ clips), 15% validation, and 15% held-out test splits.
- **Normalization Pipeline:** Temporal windows of $T = 16$ frames at 30 FPS. Each frame contains 21 3D landmarks $P_t \in \mathbb{R}^{21 \times 3}$. To guarantee scale and position invariance:
  1. *Wrist Translation Invariance:* $\tilde{p}_{t,i} = p_{t,i} - p_{t,0} \quad \forall i \in \{0, \dots, 20\}$.
  2. *Anatomical Scale Normalization:* $x_{t,i} = \tilde{p}_{t,i} / (\|\tilde{p}_{t,9}\|_2 + \epsilon)$ using the wrist-to-MCP distance.
  Input tensor: $X \in \mathbb{R}^{B \times 63 \times 16}$.

---

### 3. Baseline Model & Success Metrics
- **Model Architecture:** 1D Conv input projection $\rightarrow$ 3 Depthwise-Separable Temporal Residual Blocks with Squeeze-and-Excitation attention $\rightarrow$ Global Average Pooling $\rightarrow$ Linear classifier ($128 \rightarrow 10$ logits). Baseline size: $\approx 185\text{k}$ parameters ($740\text{ KB}$ in FP32).
- **Target Metrics:** Test Top-1 Accuracy $\ge \mathbf{92.0\%}$, Macro-F1 $\ge \mathbf{0.90}$, and zero false activations on background hand motion.

---

### 4. Edge Deployment Constraints & Budgets
Evaluated on **Apple Silicon (M-series / ANE)** and **Raspberry Pi 4 (ARM Cortex-A72, 1.5 GHz)**:

| Metric / Dimension | Specification Budget | EdgeGesture-Lite Target | Safety Margin |
|---|:---:|:---:|:---:|
| **Inference Latency** | **$\le \mathbf{50.0\text{ ms}}$** | **$\le \mathbf{12.5\text{ ms}}$** (Model) / $\le 35.0\text{ ms}$ (End-to-End) | $4.0\times$ margin |
| **Model Storage Size** | **$\le \mathbf{5.0\text{ MB}}$** | **$\le \mathbf{0.85\text{ MB}}$** (INT8 uncompressed: $185\text{ KB}$) | $5.8\times$ margin |
| **Working Memory (RAM)** | $\le 100\text{ MB}$ RSS | $\le 28.5\text{ MB}$ RSS peak working set | $3.5\times$ margin |
| **Compute / Power** | $\le 5.0\text{ W}$ Edge Budget | $\le 1.8\text{ W}$ continuous CPU dissipation | Compliant |

---

### 5. Planned Optimization Strategy
1. **INT8 Quantization (PTQ & QAT):** Symmetric per-channel weight and histogram-calibrated activation quantization via `torch.ao.quantization` and ONNX Runtime:
   $$q = \text{clamp}\left( \left\lfloor \frac{w}{S} \right\rceil + Z, -128, 127 \right)$$
   Yields $4\times$ model size reduction ($740\text{ KB} \rightarrow 185\text{ KB}$) and $2.8\times$ speedup via ARM NEON vector instructions.
2. **Knowledge Distillation:** A Spatio-Temporal Graph ConvNet (ST-GCN, 3.2M params) acts as teacher to transfer structural representations to the lightweight student TCN using temperature-scaled cross-entropy and KL-divergence loss.
3. **Structured Pruning:** 30% channel pruning based on $\ell_1$-norm ranking to optimize memory bandwidth and cache locality.

---

### 6. Roadmap
- **Weeks 1–3:** Data ingestion, landmark normalization, and FP32 baseline training.
- **Weeks 4–6:** ST-GCN teacher training and Knowledge Distillation transfer.
- **Weeks 7–8:** INT8 PTQ/QAT quantization, ONNX Runtime conversion, latency/memory profiling.
- **Weeks 9–10:** Verification against edge budgets ($\le 50\text{ ms}, \le 5\text{ MB}$), benchmark report, and interactive demo.
