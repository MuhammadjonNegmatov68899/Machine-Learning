# Lab 3: Architectures and Their Costs — Comprehensive Technical Report

**Course**: Machine Learning  
**Author**: Muhammadjon Negmatov  
**Target Environment**: Python 3.11.8, PyTorch 2.3.1, Apple Silicon MPS & CPU  
**Dataset**: Fashion-MNIST (`torchvision.datasets.FashionMNIST`)  

---

## 1. Executive Summary & Experimental Objectives

The purpose of this laboratory is to systematically evaluate, profile, and compare the three fundamental deep neural network architectural paradigms:
1. **Multi-Layer Perceptrons (MLPs)** — dense, fully-connected networks without spatial priors.
2. **Convolutional Neural Networks (CNNs)** — networks leveraging weight sharing and translation equivariance.
3. **Patch-based Vision Transformers (Attention)** — sequence-to-sequence networks relying on multi-head self-attention.

We examine each architecture along three primary dimensions:
- **Model Capacity & Parameter Costs**: Theoretical parameter count derivations compared against automated summaries (`torchinfo.summary`).
- **Computational Complexity**: Theoretical Multiply-Accumulate (MAC) and Floating-Point Operations (FLOPs) using the formulas from Reddi Chapter 4, contrasted against profiler output (`thop.profile`) and tool limitations.
- **Inference Latency & Throughput**: Hardware execution timing across batch size $1$ (latency-critical single-item inference) and batch size $128$ (throughput-oriented batch inference) on both Apple Silicon MPS and CPU.
- **Empirical Generalization**: Classification performance on Fashion-MNIST trained for exactly 5 epochs using Adam ($\text{lr} = 10^{-3}$, batch size $128$).
- **Architectural & System-Level Trade-offs**: Memory-access patterns, arithmetic intensity, memory bandwidth, and spatial cache reuse (Reddi §4.6–§4.7).

---

## 2. Dataset and Preprocessing Protocol

### 2.1 Dataset Specifications
- **Dataset**: Fashion-MNIST, composed of $28 \times 28$ grayscale images across 10 clothing categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- **Global Normalization**:
  $$\mu = 0.2860, \quad \sigma = 0.3530$$
  Each pixel $x_{i,j} \in [0, 1]$ produced by `transforms.ToTensor()` is standardized via:
  $$x_{\text{norm}} = \frac{x - 0.2860}{0.3530}$$
  These values represent the dataset-wide mean and standard deviation across the Fashion-MNIST training corpus, ensuring zero-centered inputs with unit variance to optimize gradient flow during early Adam updates.

### 2.2 Data Partitioning and Reproducibility
- **Official Training Partition**: 60,000 images partitioned into:
  - **Training Subset**: 55,000 samples ($430$ mini-batches of size $128$).
  - **Validation Subset**: 5,000 samples ($40$ mini-batches of size $128$).
  - Splitting was performed deterministically using:
    ```python
    torch.utils.data.random_split(
        full_train_dataset, [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )
    ```
- **Official Test Partition**: 10,000 images ($79$ mini-batches of size $128$), strictly held out and evaluated only once after the completion of 5 training epochs. The test set was never used for model selection, early stopping, or learning rate tuning.
- **Global Seed**: Seed `42` was explicitly set across Python's `random`, `numpy`, and PyTorch (`manual_seed` and `mps.manual_seed`) to ensure exact reproducibility across multiple runs.

---

## 3. Model Architectures & Parameter Count Analysis

### 3.1 Mathematical Formulations and Parameter Derivations

#### 1. Multi-Layer Perceptron (MLP)
- **Specification**: Flatten $784 \to \text{Linear}(784, 256) \to \text{ReLU} \to \text{Linear}(256, 128) \to \text{ReLU} \to \text{Linear}(128, 10)$.
- **Analytical Parameter Count**:
  $$\text{Params}_{\text{Linear}} = \text{in\_features} \times \text{out\_features} + \text{out\_features}$$
  - $\text{Layer 1 } (784 \to 256): 784 \times 256 + 256 = 200{,}704 + 256 = 200{,}960$
  - $\text{Layer 2 } (256 \to 128): 256 \times 128 + 128 = 32{,}768 + 128 = 32{,}896$
  - $\text{Layer 3 } (128 \to 10): 128 \times 10 + 10 = 1{,}280 + 10 = 1{,}290$
  - **Total Parameters**: $200{,}960 + 32{,}896 + 1{,}290 = \mathbf{235{,}146}$.

#### 2. Convolutional Neural Network (CNN)
- **Specification**: $\text{Conv2d}(1, 16, 3, \text{pad}=1) \to \text{ReLU} \to \text{MaxPool2d}(2) \to \text{Conv2d}(16, 32, 3, \text{pad}=1) \to \text{ReLU} \to \text{MaxPool2d}(2) \to \text{Flatten} \to \text{Linear}(32 \times 7 \times 7, 10)$.
- **Spatial Tensor Dimensions**:
  - Input: $(B, 1, 28, 28)$
  - Conv1 + ReLU: $(B, 16, 28, 28)$
  - MaxPool1: $(B, 16, 14, 14)$
  - Conv2 + ReLU: $(B, 32, 14, 14)$
  - MaxPool2: $(B, 32, 7, 7)$
  - Flatten: $(B, 1568)$ where $1568 = 32 \times 7 \times 7$
  - Linear: $(B, 10)$
- **Analytical Parameter Count**:
  $$\text{Params}_{\text{Conv2d}} = K \times K \times C_{\text{in}} \times C_{\text{out}} + C_{\text{out}}$$
  - $\text{Conv1 } (1 \to 16, 3\times 3): 3 \times 3 \times 1 \times 16 + 16 = 144 + 16 = 160$
  - $\text{Conv2 } (16 \to 32, 3\times 3): 3 \times 3 \times 16 \times 32 + 32 = 4{,}608 + 32 = 4{,}640$
  - $\text{Linear } (1568 \to 10): 1{,}568 \times 10 + 10 = 15{,}680 + 10 = 15{,}690$
  - **Total Parameters**: $160 + 4{,}640 + 15{,}690 = \mathbf{20{,}490}$.

#### 3. Patch-Based Vision Transformer (Attention)
- **Specification**:
  - Image split: $28 \times 28$ image into sixteen $7 \times 7$ non-overlapping patches ($4 \times 4$ grid, sequence length $S = 16$).
  - Linear patch projection to dimension $d_{\text{model}} = 64$ via `nn.Conv2d(1, 64, kernel_size=7, stride=7)`.
  - Learnable 1D position embeddings: tensor of shape $(1, 16, 64)$.
  - 2 Transformer encoder layers: $d_{\text{model}} = 64, n_{\text{head}} = 4, d_{\text{ff}} = 128, \text{activation} = \text{ReLU}$.
  - Sequence mean-pooling across the 16 tokens: $(B, 16, 64) \to (B, 64)$.
  - Classification head: $\text{Linear}(64, 10)$.
- **Analytical Parameter Count**:
  - Patch projection: $7 \times 7 \times 1 \times 64 + 64 = 3{,}136 + 64 = 3{,}200$
  - Learnable Position Embeddings: $16 \times 64 = 1{,}024$
  - Each Transformer Encoder Layer:
    - Multi-Head Attention (MHA):
      - Q, K, V Projections: $3 \times (64 \times 64 + 64) = 3 \times 4{,}160 = 12{,}480$
      - Output Projection: $64 \times 64 + 64 = 4{,}160$
      - Total MHA params = $16{,}640$
    - Feed-Forward Network (FFN):
      - Linear 1 ($64 \to 128$): $64 \times 128 + 128 = 8{,}320$
      - Linear 2 ($128 \to 64$): $128 \times 64 + 64 = 8{,}256$
      - Total FFN params = $16{,}576$
    - Layer Normalization (2 layers per block, affine parameters $\gamma, \beta$):
      - $2 \times (2 \times 64) = 256$
    - Total per Encoder Layer: $16{,}640 + 16{,}576 + 256 = 33{,}472$
  - 2 Transformer Layers: $2 \times 33{,}472 = 66{,}944$
  - Linear Head ($64 \to 10$): $64 \times 10 + 10 = 650$
  - **Total Parameters**: $3{,}200 + 1{,}024 + 66{,}944 + 650 = \mathbf{71{,}818}$ (or $70{,}794$ without position embeddings).

---

### 3.2 Target Capacity Discrepancy & Calibrated Architectures

The lab manual states: *"Define three models of similar size, target 100k–150k parameters"*, yet the explicit architectural layer descriptions yield:
- **MLP**: $235{,}146$ parameters ($+56.8\%$ above the $150\text{k}$ ceiling).
- **CNN**: $20{,}490$ parameters ($-79.5\%$ below the $100\text{k}$ floor).
- **Attention**: $71{,}818$ parameters ($-28.2\%$ below the $100\text{k}$ floor).

Rather than silently altering the models or violating the layer definitions in the manual, our solution implements **both**:
1. **`specified` mode**: Preserves the exact layer dimensions from the manual for rigorous verification and automated grading.
2. **`calibrated` mode**: Applies the least disruptive, mathematically justified adjustments to bring each architecture inside the $100{,}000$–$150{,}000$ parameter target window while preserving topology:
   - **Calibrated MLP**: Scales hidden dimensions proportionally from $(256, 128)$ to $(144, 72)$.
     $$\text{Params} = (784 \times 144 + 144) + (144 \times 72 + 72) + (72 \times 10 + 10) = \mathbf{124{,}210}$$
   - **Calibrated CNN**: Preserves the 2 conv + 2 maxpool + linear topology, scaling filter channels from $(16, 32)$ to $(64, 128)$.
     $$\text{Conv1: } 1 \times 64 \times 9 + 64 = 640$$
     $$\text{Conv2: } 64 \times 128 \times 9 + 128 = 73{,}856$$
     $$\text{Linear: } (128 \times 7 \times 7) \times 10 + 10 = 6{,}272 \times 10 + 10 = 62{,}730$$
     $$\text{Total Params} = 640 + 73{,}856 + 62{,}730 = \mathbf{137{,}226}$$
   - **Calibrated Attention**: Retains identical patch projection ($64$), attention heads ($4$), and feed-forward dimension ($128$), but increases the number of Transformer encoder layers from $2$ to $4$.
     $$\text{Total Params} = 3{,}200 + 1{,}024 + 4 \times 33{,}472 + 650 = \mathbf{138{,}762}$$

All 6 models were fully trained, benchmarked, and evaluated under identical conditions.

---

## 4. Computational Cost: Hand Calculations vs. Profilers

### 4.1 FLOPs vs. MACs Counting Conventions
- **Multiply-Accumulate Operation (MAC)**: Computes $a \leftarrow a + (b \times c)$. A single MAC represents **1 multiply and 1 add**, which equals **2 Floating-Point Operations (FLOPs)** in standard numerical analysis and hardware roofline models.
- **Profiling Conventions**:
  - `torchinfo` reports `mult_adds` (which equals MACs).
  - `thop` measures MACs (`total_ops`), although profiler literature often casually terms this "FLOPs".
  - Chapter 4 in Reddi uses the explicit 2-FLOPs-per-MAC convention: $\text{Linear} = 2 \times \text{in} \times \text{out}$, $\text{Conv2d} = 2 \times K \times K \times C_{\text{in}} \times C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}$.

### 4.2 Representative Layer Calculations

| Layer / Component | Configuration | Parameters | Hand MACs | Hand FLOPs ($2 \times \text{MACs}$) | Analytical Formula |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Linear** | MLP Layer 1 ($784 \to 256$) | $200{,}960$ | $200{,}704$ | $401{,}408$ | $\text{MACs} = \text{in} \times \text{out}, \text{FLOPs} = 2 \times \text{in} \times \text{out}$ |
| **Conv2d** | CNN Conv 2 ($16 \to 32, 3\times 3, 14\times 14$) | $4{,}640$ | $903{,}168$ | $1{,}806{,}336$ | $\text{MACs} = K^2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}}$ |
| **Attention: QKV Projections** | $3 \times (16 \times 64 \to 16 \times 64)$ | $12{,}480$ | $196{,}608$ | $393{,}216$ | $\text{MACs} = 3 \cdot S \cdot d^2, \text{FLOPs} = 6 \cdot S \cdot d^2$ |
| **Attention: Scores ($QK^T$)** | $(16 \times 64) \times (64 \times 16) \to 16 \times 16$ | $0$ | $16{,}384$ | $32{,}768$ | $\text{MACs} = S^2 \cdot d, \text{FLOPs} = 2 \cdot S^2 \cdot d$ |
| **Attention: Context ($AV$)** | $(16 \times 16) \times (16 \times 64) \to 16 \times 64$ | $0$ | $16{,}384$ | $32{,}768$ | $\text{MACs} = S^2 \cdot d, \text{FLOPs} = 2 \cdot S^2 \cdot d$ |
| **Attention: Output Proj ($W_O$)** | $(16 \times 64) \times (64 \times 64) \to 16 \times 64$ | $4{,}160$ | $65{,}536$ | $131{,}072$ | $\text{MACs} = S \cdot d^2, \text{FLOPs} = 2 \cdot S \cdot d^2$ |
| **Attention: MHA Block Total** | Sequence $S=16$, Dimension $d=64$ | $16{,}640$ | $294{,}912$ | $589{,}824$ | $\text{MACs} = 4Sd^2 + 2S^2d, \text{FLOPs} = 8Sd^2 + 4S^2d$ |
| **Attention: FFN Block** | Linear 1 ($64 \to 128$) + Linear 2 ($128 \to 64$) | $16{,}576$ | $262{,}144$ | $524{,}288$ | $\text{MACs} = 2 \cdot S \cdot d \cdot d_{\text{ff}}, \text{FLOPs} = 4 \cdot S \cdot d \cdot d_{\text{ff}}$ |
| **Attention: Full Encoder Layer** | MHA Block + FFN Block + 2 LayerNorms | $33{,}472$ | $557{,}056$ | $1{,}114{,}112$ | $\text{MACs} = \text{MHA}_{\text{macs}} + \text{FFN}_{\text{macs}}$ |

#### Step-by-Step Derivation of Transformer Encoder Layer FLOPs ($S=16, d=64, d_{\text{ff}}=128$):
1. **Q, K, V Projections**: 3 linear projections of input $X \in \mathbb{R}^{16 \times 64}$ by weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{64 \times 64}$.
   $$\text{MACs} = 3 \times (16 \times 64 \times 64) = 196{,}608 \implies \mathbf{393{,}216 \text{ FLOPs}}$$
2. **Attention Scores ($QK^T$)**: Dynamic matrix multiplication between Query ($16 \times 64$) and Key transpose ($64 \times 16$).
   $$\text{MACs} = 16 \times 16 \times 64 = 16{,}384 \implies \mathbf{32{,}768 \text{ FLOPs}}$$
3. **Attention Context ($A \times V$)**: Dynamic matrix multiplication between normalized Attention weights ($16 \times 16$) and Value ($16 \times 64$).
   $$\text{MACs} = 16 \times 16 \times 64 = 16{,}384 \implies \mathbf{32{,}768 \text{ FLOPs}}$$
4. **Output Linear Projection ($W_O$)**: Projection of concatenated context matrix ($16 \times 64$) through $W_O \in \mathbb{R}^{64 \times 64}$.
   $$\text{MACs} = 16 \times 64 \times 64 = 65{,}536 \implies \mathbf{131{,}072 \text{ FLOPs}}$$
5. **Feed-Forward Network (FFN)**: Two consecutive dense transformations across all 16 tokens:
   - Linear 1 ($64 \to 128$): $16 \times 64 \times 128 = 131{,}072$ MACs ($262{,}144$ FLOPs).
   - Linear 2 ($128 \to 64$): $16 \times 128 \times 64 = 131{,}072$ MACs ($262{,}144$ FLOPs).
   $$\text{FFN MACs} = 131{,}072 + 131{,}072 = 262{,}144 \implies \mathbf{524{,}288 \text{ FLOPs}}$$
6. **Total per Transformer Encoder Layer**:
   $$\text{Layer MACs} = 294{,}912 + 262{,}144 = \mathbf{557{,}056 \text{ MACs}}$$
   $$\text{Layer FLOPs} = 2 \times 557{,}056 = \mathbf{1{,}114{,}112 \text{ FLOPs}}$$

---

### 4.3 Profiler Comparison and Tool Discrepancy Analysis

The table below contrasts our hand-calculated values with the measurements produced by `torchinfo.summary`, native unhooked `thop.profile`, and our verified custom-hooked `thop` implementation:

| Architecture | Mode | Hand Params | torchinfo Params | thop Raw Params | Hand MACs | Hand FLOPs | torchinfo Mult-Adds | thop Native MACs (Raw) | thop Custom MACs (Verified) | Effective Full Model FLOPs |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | Specified | $235{,}146$ | $235{,}146$ | $235{,}146$ | $234{,}752$ | $469{,}504$ | $235{,}146$ | $234{,}752$ | $234{,}752$ | $469{,}504$ |
| **CNN** | Specified | $20{,}490$ | $20{,}490$ | $20{,}490$ | $1{,}031{,}744$ | $2{,}063{,}488$ | $1{,}050{,}570$ | $1{,}031{,}744$ | $1{,}031{,}744$ | $2{,}063{,}488$ |
| **Attention** | Specified | $71{,}818$ | $71{,}818$ | **$3{,}850$** | $1{,}164{,}928$ | $2{,}329{,}856$ | **$51{,}850$** | **$50{,}816$** | **$1{,}164{,}928$** | **$2{,}329{,}856$** |
| **MLP** | Calibrated | $124{,}210$ | $124{,}210$ | $124{,}210$ | $123{,}984$ | $247{,}968$ | $124{,}210$ | $123{,}984$ | $123{,}984$ | $247{,}968$ |
| **CNN** | Calibrated | $137{,}226$ | $137{,}226$ | $137{,}226$ | $14{,}964{,}992$ | $29{,}929{,}984$ | $15{,}040{,}266$ | $14{,}964{,}992$ | $14{,}964{,}992$ | $29{,}929{,}984$ |
| **Attention** | Calibrated | $138{,}762$ | $138{,}762$ | **$3{,}850$** | $2{,}279{,}040$ | $4{,}558{,}080$ | **$51{,}850$** | **$50{,}816$** | **$2{,}279{,}040$** | **$4{,}558{,}080$** |

#### Why Profiler Outputs Differ:
1. **MACs vs. FLOPs Convention**: `thop` counts Multiply-Accumulate operations (MACs). Hand-calculated FLOPs are exactly $2 \times \text{thop MACs}$.
2. **Bias Addition Handling**:
   - In MLP (specified), weight multiplication requires $784 \times 256 + 256 \times 128 + 128 \times 10 = 234{,}752$ MACs. `thop` measures exactly $234{,}752$.
   - `torchinfo` counts $235{,}146$, adding the $394$ bias vector additions ($234{,}752 + 256 + 128 + 10 = 235{,}146$).
   - In CNN (specified), `thop` measures $1{,}031{,}744$ MACs, while `torchinfo` reports $1{,}050{,}570$ (accounting for bias additions across feature map activations: $16 \times 28^2 + 32 \times 14^2 + 10 = 12{,}544 + 6{,}272 + 10 = 18{,}826$; $1{,}031{,}744 + 18{,}826 = 1{,}050{,}570$).
3. **The Attention Model Discrepancy (Critical Finding)**:
   - `thop` reports only **$3{,}850$ parameters** and **$50{,}816$ MACs** for Attention!
   - This occurs because `thop` **lacks profiling hooks for PyTorch's native `nn.TransformerEncoderLayer` and `nn.MultiheadAttention`**.
   - As a result, `thop` completely ignores both Transformer encoder blocks. It only profiles:
     - The patch projection conv: $49 \times 1 \times 64 \times 16 = 50{,}176$ MACs and $3{,}200$ params.
     - The classification head: $64 \times 10 = 640$ MACs and $650$ params.
     - Total profiled = $50{,}176 + 640 = \mathbf{50{,}816 \text{ MACs}}$ and $3{,}200 + 650 = \mathbf{3{,}850 \text{ params}}$.
   - `torchinfo` correctly computes the $71{,}818$ total parameters, but its mult-add hooks also skip the attention matrix operations, capturing only $51{,}850$ mult-adds.
   - **Conclusion**: Relying solely on profiler packages like `thop` for Transformer architectures produces severe undercounting (by a factor of $\approx 23\times$). Rigorous hand calculations are necessary for accurate complexity analysis.

### 4.4 Handwritten Calculations Document

The complete handwritten step-by-step mathematical derivations for the MLP, CNN, and Multi-Head Attention layers—covering parameter counts, MACs, and FLOPs according to the textbook formulas—are compiled and archived in the accompanying document:

👉 **[Handwritten Calculations Document (PDF)](hand_calculations/hand_calculations.pdf)**

This document contains two upright, legible pages:
- **Page 1**: Analytical parameter counts, MACs, and FLOPs for the Multilayer Perceptron (MLP) linear layers, as well as the Convolutional Neural Network (CNN) feature dimensions and Conv1/Conv2 parameter counts.
- **Page 2**: CNN MACs and FLOPs completion, alongside the full step-by-step derivation for the Multi-Head Attention (MHA) and Feed-Forward Network (FFN) blocks inside the Transformer Encoder layer ($QKV$ projections, attention scores $QK^T$, attention context $AV$, output projection $W_O$, and dual linear feed-forward expansions).

---

## 5. Inference Latency Benchmarking

### 5.1 Protocol and Synchronization
- **Execution Mode**: Evaluated strictly inside `torch.inference_mode()`, disabling autograd tracking and intermediate graph constructions.
- **Warm-Up Runs**: 15 untimed warmup passes for batch 1; 10 warmup passes for batch 128 to prime GPU shaders and memory pipelines.
- **Repeatability**:
  - **Batch 1**: 100 timed iterations, reporting median, mean, and standard deviation.
  - **Batch 128**: 20 timed iterations, reporting median, mean, and standard deviation.
- **Hardware Synchronization**: Apple Silicon MPS requires explicit synchronization (`torch.mps.synchronize()`) before and after the timing block (`time.perf_counter()`). Omitting synchronization measures only the asynchronous kernel launch time rather than actual GPU execution.
- **Dual-Device Baseline**: Evaluated on both Apple Silicon MPS (GPU accelerator) and CPU to contrast parallel throughput against low-latency single-thread execution.

### 5.2 Measured Latency and Throughput Results

| Architecture | Mode | Device | Batch 1 Median (ms) | Batch 1 Std (ms) | Batch 128 Median (ms) | Batch 128 Std (ms) | Per-Sample Latency (ms) | Throughput (samples/s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | Specified | **MPS** | $0.361$ | $0.161$ | $0.450$ | $0.246$ | $0.0035$ | $284{,}523$ |
| **MLP** | Specified | **CPU** | $0.027$ | $0.005$ | $0.093$ | $0.012$ | $0.0007$ | $1{,}373{,}884$ |
| **CNN** | Specified | **MPS** | $0.436$ | $0.037$ | $1.236$ | $0.041$ | $0.0097$ | $103{,}574$ |
| **CNN** | Specified | **CPU** | $0.095$ | $0.008$ | $4.928$ | $0.184$ | $0.0385$ | $25{,}976$ |
| **Attention** | Specified | **MPS** | $1.770$ | $0.147$ | $3.189$ | $0.373$ | $0.0249$ | $40{,}142$ |
| **Attention** | Specified | **CPU** | $0.261$ | $0.019$ | $6.053$ | $0.312$ | $0.0473$ | $21{,}146$ |
| **MLP** | Calibrated | **MPS** | $0.332$ | $0.186$ | $0.459$ | $0.065$ | $0.0036$ | $278{,}652$ |
| **MLP** | Calibrated | **CPU** | $0.026$ | $0.004$ | $0.072$ | $0.008$ | $0.0006$ | $1{,}788{,}645$ |
| **CNN** | Calibrated | **MPS** | $0.475$ | $0.116$ | $4.391$ | $0.065$ | $0.0343$ | $29{,}148$ |
| **CNN** | Calibrated | **CPU** | $0.270$ | $0.012$ | $19.367$ | $0.452$ | $0.1513$ | $6{,}609$ |
| **Attention** | Calibrated | **MPS** | $4.032$ | $0.331$ | $8.016$ | $0.914$ | $0.0626$ | $15{,}968$ |
| **Attention** | Calibrated | **CPU** | $0.470$ | $0.035$ | $10.939$ | $0.419$ | $0.0855$ | $11{,}701$ |

### 5.3 Key Latency Insights: CPU vs. GPU Accelerator
1. **The Batch-1 GPU Dispatch Overhead**:
   - At batch size 1, the CPU outperforms the MPS accelerator across all models (e.g., Specified MLP takes $0.027$ ms on CPU vs. $0.361$ ms on MPS; CNN takes $0.095$ ms on CPU vs. $0.436$ ms on MPS).
   - For small workloads, the CPU benefits from immediate execution with near-zero launch latency, keeping all weights resident in L1/L2 caches. Conversely, GPU execution incurs command buffer dispatch, kernel launching, and memory synchronization overheads ($\approx 0.3$–$0.4$ ms) that exceed the arithmetic execution time.
2. **Batch-128 Accelerator Saturation**:
   - At batch size 128, the massive parallel compute cores of the GPU dominate.
   - For Specified CNN, GPU execution takes only $1.236$ ms ($0.0097$ ms/sample) vs. $4.928$ ms on CPU (**4.0x speedup**).
   - For Calibrated CNN ($137\text{k}$ params, $15\text{M}$ MACs), GPU execution is $4.391$ ms vs. $19.367$ ms on CPU (**4.4x speedup**).
   - Batching increases arithmetic intensity, enabling the accelerator to hide memory latency behind parallel arithmetic throughput.

---

## 6. Training Dynamics & Empirical Performance

All models were trained for exactly 5 epochs using Adam ($\text{lr} = 10^{-3}$) and mini-batch size $128$ on Apple Silicon MPS.

### 6.1 Epoch-by-Epoch Convergence Table

| Epoch | Metric | MLP (Spec) | CNN (Spec) | Attn (Spec) | MLP (Calib) | CNN (Calib) | Attn (Calib) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Train Loss / Val Acc | $0.4919$ / $84.50\%$ | $0.5122$ / $86.52\%$ | $0.8886$ / $76.22\%$ | $0.5192$ / $85.58\%$ | $0.4141$ / $87.12\%$ | $0.7916$ / $78.24\%$ |
| **2** | Train Loss / Val Acc | $0.3540$ / $86.34\%$ | $0.3357$ / $88.34\%$ | $0.5676$ / $80.10\%$ | $0.3693$ / $86.32\%$ | $0.2759$ / $90.44\%$ | $0.5127$ / $82.28\%$ |
| **3** | Train Loss / Val Acc | $0.3134$ / $87.38\%$ | $0.2960$ / $88.78\%$ | $0.5039$ / $80.32\%$ | $0.3291$ / $87.46\%$ | $0.2302$ / $90.74\%$ | $0.4616$ / $82.72\%$ |
| **4** | Train Loss / Val Acc | $0.2888$ / $88.06\%$ | $0.2723$ / $89.56\%$ | $0.4707$ / $81.12\%$ | $0.3067$ / $87.22\%$ | $0.2016$ / $91.08\%$ | $0.4257$ / $84.52\%$ |
| **5** | Train Loss / Val Acc | $0.2696$ / $87.48\%$ | $0.2542$ / $89.64\%$ | $0.4493$ / $83.76\%$ | $0.2826$ / $87.36\%$ | $0.1795$ / $91.58\%$ | $0.3986$ / $84.58\%$ |
| **Final** | **Test Loss** | **$0.3566$** | **$0.2908$** | **$0.4718$** | **$0.3549$** | **$0.2549$** | **$0.4411$** |
| **Final** | **Test Accuracy** | **$87.39\%$** | **$89.54\%$** | **$82.87\%$** | **$87.30\%$** | **$90.89\%$** | **$84.20\%$** |

---

## 7. Visualizations and Trade-Off Plots

The experimental pipeline automatically generated three figures saved in `results/`:

### 7.1 Accuracy vs. Parameter Count
![Accuracy vs. Parameters](results/accuracy_vs_params.png)
- **Observations**:
  - The CNN achieves superior classification accuracy ($89.54\%$) at a fraction of the parameter cost ($20{,}490$ parameters) compared to the MLP ($87.39\%$ at $235{,}146$ parameters).
  - When calibrated to the $100\text{k}$–$150\text{k}$ target range, the CNN scales to **$90.89\%$ accuracy**, widening its lead over the calibrated MLP ($87.30\%$) and calibrated Vision Transformer ($84.20\%$).

### 7.2 Accuracy vs. Inference Latency Trade-Off
![Accuracy vs. Latency](results/accuracy_vs_latency.png)
- **Observations**:
  - For single-sample inference (Batch 1), MLP offers the lowest latency ($0.33$–$0.36$ ms on MPS; $0.026$–$0.027$ ms on CPU), but CNN delivers a significantly higher accuracy-to-latency ratio.
  - At batch size 128, CNN maintains an optimal frontier: $89.54\%$ accuracy in $1.24$ ms (Specified) and $90.89\%$ accuracy in $4.39$ ms (Calibrated).
  - Attention exhibits the highest inference latency ($3.19$ ms specified, $8.02$ ms calibrated on MPS) while achieving the lowest accuracy.

### 7.3 Training and Validation Dynamics
![Training Curves](results/training_curves.png)
- **Observations**:
  - Both MLP and CNN converge rapidly within the first 3 epochs.
  - The Attention model learns significantly slower, starting from a high initial loss ($0.8886$ in epoch 1) and requiring all 5 epochs to surpass $82\%$ accuracy.

---

## 8. Architectural and System-Level Trade-Offs (Reddi §4.6–§4.7)

Understanding why these three models perform so differently requires analyzing their structural inductive biases and hardware memory-access patterns as outlined in Reddi Chapter 4.

### 8.1 Inductive Bias vs. Sample Efficiency
1. **Convolutional Neural Networks (Translation Equivariance & Locality)**:
   - Images exhibit two core physical properties: **spatial locality** (neighboring pixels are highly correlated) and **translation invariance/equivariance** (a shoe is a shoe regardless of where it appears in the frame).
   - The CNN enforces these priors directly through localized $3 \times 3$ receptive fields and shared kernel weights.
   - Consequently, the CNN requires drastically fewer parameters ($20.5\text{k}$) and trains efficiently on 55,000 samples, reaching $89.54\%$ accuracy in 5 epochs.
2. **Multi-Layer Perceptrons (Complete Permutation Invariance)**:
   - Flattening the image destroys all 2D spatial coordinate relationships.
   - The MLP treats pixel $(0, 0)$ and pixel $(0, 1)$ with the same initial relationship as pixel $(0, 0)$ and pixel $(27, 27)$.
   - While it can memorize patterns across the dataset ($87.39\%$), it requires over $200\text{k}$ parameters in its first dense layer alone ($784 \times 256$), resulting in high parameter redundancy and vulnerability to spatial shifts.
3. **Vision Transformers (Weak Inductive Bias)**:
   - Self-attention operates on sets of tokens. Without convolutions, it has no intrinsic knowledge of 2D geometry; it must learn the spatial configuration of the 16 patches entirely from data via position embeddings and attention weights.
   - On small datasets like Fashion-MNIST without large-scale pretraining (e.g., ImageNet-21k), 5 epochs are insufficient for self-attention maps to specialize. Hence, Attention achieves $82.87\%$ (specified) and $84.20\%$ (calibrated).

---

### 8.2 Memory-Access Patterns and Hardware Roofline Analysis

Hardware execution speed is governed not only by peak FLOPs, but by **Operational Intensity** ($\text{FLOPs} / \text{Byte of DRAM traffic}$) and memory hierarchy traversal:

```text
  +-------------------------------------------------------------------------+
  | Memory Hierarchy:                                                       |
  | Registers / L1 Cache  (~TB/s, KBs)                                      |
  |   ^                                                                     |
  | L2 / System Cache     (~hundreds GB/s, MBs)                             |
  |   ^                                                                     |
  | Main Memory (DRAM)    (~tens to low hundreds GB/s, GBs)                 |
  +-------------------------------------------------------------------------+
```

1. **MLP: Memory-Bandwidth Bound ($I_{\text{op}} \ll 1$)**:
   - In single-item inference ($B=1$), every weight in the $784 \times 256$ matrix ($200{,}704$ floats $= 802.8$ KB) must be transferred from memory to compute registers to perform just one multiplication per weight.
   - Arithmetic intensity is approximately $1 \text{ FLOP} / \text{Byte}$.
   - The processor spends most cycles waiting on memory bus transfers rather than compute ALUs.
   - Increasing batch size to $128$ reuses the weights across all 128 samples, boosting operational intensity by $128\times$ and amortizing memory traffic.
2. **CNN: Compute-Bound with High Cache Reuse ($I_{\text{op}} \gg 1$)**:
   - In a $3 \times 3$ convolution, a compact kernel ($3 \times 3 \times C_{\text{in}} \times C_{\text{out}}$) is repeatedly applied across an entire $H \times W$ spatial grid.
   - The weights fit entirely inside L1 cache/registers and remain resident throughout the forward pass.
   - Activations enjoy 2D sliding-window spatial reuse: adjacent output pixels reuse $6$ out of $9$ input pixels from the previous step.
   - This maximizes arithmetic intensity, allowing GPUs and systolic arrays to operate near peak TFLOPs throughput.
3. **Attention: Intermediate Activation and Synchronization Heavy**:
   - While MHA projections ($W_Q, W_K, W_V, W_O$) are dense matrix multiplications, the self-attention core requires materializing the attention score matrix:
     $$A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{B \times H \times S \times S}$$
   - These intermediate tensors must be written to and read from memory, generating substantial memory traffic.
   - Furthermore, Transformer layers require multiple sequential reductions: LayerNorm (mean and variance across feature dimension), Softmax (normalization across sequence), and Residual Additions. Each reduction acts as a memory barrier preventing kernel fusion unless specialized fusion techniques (such as FlashAttention) are applied.
   - This high activation footprint and synchronization frequency explains why Attention exhibits the highest latency ($3.19$ ms specified, $8.02$ ms calibrated) on batch 128.

---

### 8.3 Final Verdict: Cost-Effectiveness Champion

Based on empirical evidence across parameters, compute cost, latency, and classification accuracy:
- **Most Cost-Effective Architecture: The Convolutional Neural Network (CNN)**.
- **Justification**:
  - **Parameter Efficiency**: Achieves $89.54\%$ accuracy with only $20{,}490$ parameters ($11.5\times$ fewer parameters than MLP; $3.5\times$ fewer than Attention).
  - **Latency and Throughput**: Executes in $1.24$ ms on GPU at batch 128, processing over $103{,}000$ images per second.
  - **Scaling Behavior**: When calibrated to $137\text{k}$ parameters, it reaches **$90.89\%$ test accuracy**, the highest across all tested models.
  - **Hardware Synergy**: Its high operational intensity maps directly to modern GPU cache hierarchies and tensor compute pipelines.

---

## 9. Verification Checklist & Compliance Summary

- [x] **Dataset**: Official Fashion-MNIST downloaded via `torchvision.datasets.FashionMNIST`, normalized by $(0.2860, 0.3530)$, split into $55\text{k}$ train, $5\text{k}$ val, and $10\text{k}$ test. Batch size 128.
- [x] **Architectures**: Implemented exact specified MLP ($235{,}146$ params), CNN ($20{,}490$ params), and Attention ($71{,}818$ params). Provided calibrated versions ($124\text{k}$–$139\text{k}$ params) addressing the manual's capacity headline.
- [x] **Compute Cost**: Hand calculations provided for representative Linear, Conv2d, and MHA layers. Automated counts verified with `torchinfo.summary`. Profiler gaps in `thop.profile` (convention gap, bias omission, missing Transformer hooks) documented in detail.
- [x] **Inference Latency**: Benchmarked on batch 1 ($100$ runs) and batch 128 ($20$ runs) using `torch.inference_mode()`, warmup iterations, and `torch.mps.synchronize()`. Reported on both MPS and CPU.
- [x] **Training Protocol**: Exactly 5 epochs with Adam ($\text{lr} = 10^{-3}$, batch size $128$). Tracked train/val loss and evaluated on test set only after training.
- [x] **Artifacts**: All CSV files (`summary_metrics.csv`, `latency_results.csv`, `profiler_comparison.csv`, `layer_hand_costs.csv`, `training_history.csv`) and figures (`accuracy_vs_params.png`, `accuracy_vs_latency.png`, `training_curves.png`) generated and saved under `lab03/results/`.
- [x] **Safety & Git Integrity**: No modifications made to `lab01/`, `lab02/`, or git repository history.
