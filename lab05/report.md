# Machine Learning Lab 05: Frameworks and Automatic Differentiation

**Author:** Muhammadjon Negmatov  
**Institution:** University Machine Learning Laboratory  
**Environment:** macOS 27.2 (Darwin arm64), Apple Silicon, Python 3.11.8, PyTorch 2.3.1  

---

## 1. Executive Summary

This report presents the design, mathematical formulation, empirical verification, and architectural analysis for **Lab 5: Frameworks and Automatic Differentiation**. The investigation consists of two primary technical components:
1. **The Custom Reverse-Mode Automatic Differentiation Engine:** A pure Python scalar autograd engine built from first principles around a dynamic directed acyclic graph (DAG) representation (`Value` node), supporting forward graph construction, reverse topological sorting, gradient accumulation, and modular neural network layers.
2. **Framework Performance and Hardware Efficiency:** An empirical study of PyTorch computational graph tracing, eager versus compiled/traced execution modes, and the hardware micro-architectural consequences of tensor contiguity and memory layout strides.

### Key Measured Findings (Synchronized with Results CSVs)

| Evaluation Dimension | Metric / Target | Empirical Result | Verdict / Ratio |
|---|---|---|---|
| **Gradient Verification (Scalar)** | Target: `((a*b + c)**2).relu().sum()` | Max Abs Error vs PyTorch: **`0.000`**<br>Max Abs Error vs Num Diff: **`6.875e-09`** | **PASSED** (Tolerance $< 10^{-6}$) |
| **Gradient Verification (Vector)** | 5-Dimensional Vector Evaluation | Max Abs Error vs PyTorch: **`0.000`**<br>Max Abs Error vs Num Diff: **`7.430e-09`** | **PASSED** (Tolerance $< 10^{-6}$) |
| **make_moons 5-Run Median Loss** | $2 \rightarrow 8 \rightarrow 1$ MLP (150 epochs) | Custom: **`0.3426`** (Train) / **`0.2663`** (Test)<br>PyTorch: **`0.3450`** (Train) / **`0.2688`** (Test) | **Parity Achieved** ($\Delta \le 0.0026$) |
| **make_moons 5-Run Median Accuracy** | Binary Cross-Entropy, $\eta = 0.5$ | Custom: **`81.33%`** (Train) / **`90.00%`** (Test)<br>PyTorch: **`81.33%`** (Train) / **`90.00%`** (Test) | **Identical** ($\Delta = 0.00\%$) |
| **Training Latency (150 Epochs)** | Full-batch GD (150 samples) | Custom: **`9.486 s`**<br>PyTorch: **`0.018 s`** | PyTorch is **`531.1x`** faster |
| **PyTorch Inference Latency** | Batch=32, Median Latency | Eager: **`9.96 µs`**<br>Traced (`torch.jit.trace`): **`7.13 µs`**<br>Scripted (`torch.jit.script`): **`6.12 µs`** | Traced is **`1.40x`** faster<br>Scripted is **`1.63x`** faster |
| **Tensor Layout Contiguity** | Sum dim=1 on $2048 \times 2048$ matrix | Contiguous (`stride=[2048, 1]`): **`0.182 ms`**<br>Transposed (`stride=[1, 2048]`): **`0.495 ms`** | Non-contiguous is **`2.73x`** slower |

---

## 2. Reverse-Mode Automatic Differentiation Engine

### 2.1 Graph Formulation and Node Architecture

Reverse-mode automatic differentiation operates on a directed acyclic graph (DAG) where nodes represent intermediate scalar values and directed edges represent functional dependencies. In `src/engine.py`, every scalar is modeled by the `Value` class with the following core attributes:
- `data`: Stores the 64-bit floating-point forward evaluation result.
- `grad`: Stores the accumulated adjoint $\bar{v}_i = \frac{\partial \mathcal{L}}{\partial v_i}$.
- `_prev`: Set of direct ancestor (parent) nodes that generated this value.
- `_op`: String identifier of the generating operator.
- `_backward`: Local derivative closure that applies the multivariate chain rule to propagate incoming adjoints $\bar{v}_{\text{out}}$ to predecessors.

```
       [ a ]        [ b ]
         \            /
          \          /  ( * )
           v        v
          [ a * b ]       [ c ]
               \            /
                \          /  ( + )
                 v        v
              [ a * b + c ]
                    |
                    | ( ** 2 )
                    v
            [ (a * b + c) ** 2 ]
                    |
                    | ( ReLU )
                    v
                  [ out ]
```

### 2.2 Mathematical Operators and Local Gradient Closures

For every elementary function $f: \mathbb{R}^k \rightarrow \mathbb{R}$, reverse-mode AD computes vector-Jacobian products (VJPs). The table below details the mathematical definitions and local backward implementations:

| Operation | Mathematical Form | Local Gradient Rule | Closure Implementation |
|---|---|---|---|
| **Addition** | $z = x + y$ | $\frac{\partial z}{\partial x} = 1, \quad \frac{\partial z}{\partial y} = 1$ | `x.grad += 1.0 * out.grad`<br>`y.grad += 1.0 * out.grad` |
| **Multiplication** | $z = x \cdot y$ | $\frac{\partial z}{\partial x} = y, \quad \frac{\partial z}{\partial y} = x$ | `x.grad += y.data * out.grad`<br>`y.grad += x.data * out.grad` |
| **Power** | $z = x^p$ | $\frac{\partial z}{\partial x} = p \cdot x^{p-1}$ | `x.grad += (p * (x.data ** (p - 1.0))) * out.grad` |
| **ReLU** | $z = \max(0, x)$ | $\frac{\partial z}{\partial x} = \mathbf{1}_{\{x > 0\}}$ | `x.grad += (1.0 if x.data > 0 else 0.0) * out.grad` |
| **Sigmoid** | $z = \sigma(x) = \frac{1}{1 + e^{-x}}$ | $\frac{\partial z}{\partial x} = \sigma(x)(1 - \sigma(x))$ | `x.grad += (out.data * (1.0 - out.data)) * out.grad` |
| **Natural Log** | $z = \ln(x)$ | $\frac{\partial z}{\partial x} = \frac{1}{x}$ | `x.grad += (1.0 / max(x.data, eps)) * out.grad` |
| **Division** | $z = \frac{x}{y}$ | $\frac{\partial z}{\partial x} = \frac{1}{y}, \quad \frac{\partial z}{\partial y} = -\frac{x}{y^2}$ | `x.grad += (1.0 / y.data) * out.grad`<br>`y.grad += (-x.data / (y.data**2)) * out.grad` |

### 2.3 Topological Sorting and Gradient Accumulation

The reverse pass evaluates the chain rule in reverse topological order:
$$\bar{v}_i = \sum_{j \in \text{children}(i)} \bar{v}_j \frac{\partial v_j}{\partial v_i}$$

To guarantee mathematical correctness on arbitrary DAGs (including diamond dependencies and shared parameters):
1. **Iterative Post-Order DFS:** Topological sorting is constructed iteratively using an explicit call stack. This eliminates Python recursion depth limits when processing graphs containing tens of thousands of nodes (e.g., full-batch training across 150 samples).
2. **Gradient Accumulation (`+=`):** If a variable has fan-out $> 1$ (e.g., $y = x + x$ or shared neural network weight tensors across training samples), assigning `grad = ...` would overwrite incoming gradients from other branches. Accumulating via `grad += ...` guarantees that all downstream paths contribute their exact partial derivatives.

---

## 3. Task 4: Empirical Gradient Verification

### 3.1 Target Expression Analytical Derivation

The assignment mandates verification of the composite scalar expression:
$$\mathcal{L}(a, b, c) = \left((a \cdot b + c)^2\right).\text{relu}().\text{sum}()$$

Let the forward intermediate variables be:
1. $u = a \cdot b + c$
2. $v = u^2$
3. $w = \text{relu}(v) = \max(0, v)$
4. $\mathcal{L} = \text{sum}(w) = w$

Applying the chain rule backward from $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1.0$:
$$\frac{\partial \mathcal{L}}{\partial w} = 1.0$$
$$\frac{\partial \mathcal{L}}{\partial v} = \frac{\partial \mathcal{L}}{\partial w} \cdot \mathbf{1}_{\{v > 0\}} = 1.0 \quad (\text{for } v > 0)$$
$$\frac{\partial \mathcal{L}}{\partial u} = \frac{\partial \mathcal{L}}{\partial v} \cdot (2u) = 2(a \cdot b + c)$$
$$\frac{\partial \mathcal{L}}{\partial a} = \frac{\partial \mathcal{L}}{\partial u} \cdot b = 2(a \cdot b + c) \cdot b$$
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial u} \cdot a = 2(a \cdot b + c) \cdot a$$
$$\frac{\partial \mathcal{L}}{\partial c} = \frac{\partial \mathcal{L}}{\partial u} \cdot 1 = 2(a \cdot b + c)$$

For the reference input values $a = 2.0, b = -3.0, c = 10.0$:
- $u = 2(-3) + 10 = 4.0$
- $v = 4.0^2 = 16.0$
- $w = \max(0, 16.0) = 16.0$
- $\nabla_a = 2(4)(-3) = \mathbf{-24.0}$
- $\nabla_b = 2(4)(2) = \mathbf{16.0}$
- $\nabla_c = 2(4)(1) = \mathbf{8.0}$

### 3.2 Dual Reference Comparison Methodology

The custom engine gradients were compared against two independent ground-truth sources:
1. **PyTorch Autograd:** Evaluated using double precision (`torch.float64`) to eliminate single-precision truncation.
2. **Central Finite Differences:** Computed with symmetric step size $\epsilon = 1.0 \times 10^{-6}$:
   $$g_{\text{num}}(x) = \frac{\mathcal{L}(x + \epsilon) - \mathcal{L}(x - \epsilon)}{2\epsilon}$$
   The central difference truncation error is $\mathcal{O}(\epsilon^2) \approx 10^{-12}$, making it an ideal numerical sanity check.

### 3.3 Scalar and Vector Verification Results (from `gradient_comparison.csv`)

```text
Scalar Evaluation Results:
Variable  Input  Custom Grad  PyTorch Grad  Num Grad   Abs Err (PyTorch)  Abs Err (Num)  Status
a           2.0        -24.0         -24.0     -24.0        0.0000000000   1.086221e-09  PASSED
b          -3.0         16.0          16.0      16.0        0.0000000000   2.236447e-09  PASSED
c          10.0          8.0           8.0       8.0        0.0000000000   6.875382e-09  PASSED
```

- **Maximum Absolute Error vs PyTorch:** $\mathbf{0.000 \times 10^0}$ (exact bit-level floating-point match).
- **Maximum Absolute Error vs Numerical Finite Difference:** $\mathbf{6.875 \times 10^{-9}}$ (well below the required $1.0 \times 10^{-6}$ threshold).
- **Vector Evaluation (5 Dimensions):** Maximum absolute error across all 15 elements ($a_i, b_i, c_i$) was $\mathbf{7.430 \times 10^{-9}}$.

---

## 4. Task 5: Neural Network Training on `make_moons`

### 4.1 Architecture and Experimental Protocol

To prove that the custom engine genuinely performs end-to-end gradient-based optimization:
- **Dataset:** `sklearn.datasets.make_moons(n_samples=200, noise=0.20, random_state=42)`.
- **Leakage-Free Partitioning:** 75% train (150 samples) / 25% test (50 samples), stratified by class label. A `StandardScaler` was fitted strictly on `X_train` and transformed `X_test`.
- **Network Architecture ($2 \rightarrow 8 \rightarrow 1$):**
  - Input layer: 2 features.
  - Hidden layer: 8 neurons, Sigmoid activation (16 weights + 8 biases = 24 parameters).
  - Output layer: 1 neuron, Sigmoid activation (8 weights + 1 bias = 9 parameters).
  - Total parameters: **33 trainable scalar nodes**.
- **Loss Function:** Mean Binary Cross-Entropy with numerical stability safeguard $\epsilon = 10^{-15}$:
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
- **Optimization:** Full-batch gradient descent with learning rate $\eta = 0.5$ for 150 epochs.
- **Initialization Protocol:** Both models use the same Xavier uniform parameter distribution formula ($\mathcal{U}(-\sqrt{6/(n_{in}+1)}, \sqrt{6/(n_{in}+1)})$) with zero bias across the 5 seeds, instantiated via their respective native random number generators (`numpy.random.default_rng(seed)` for Custom Engine and `torch.manual_seed(seed)` for PyTorch). They therefore draw separate random sample realizations with the same distribution, rather than identical numerical weight matrices. (Exact weight synchronization is tested separately in unit tests).

### 4.2 Empirical Results Across 5 Random Seeds (from `training_comparison.csv`)

| Run ID | Seed | Engine | Epochs | Train Loss | Test Loss | Train Acc | Test Acc | Train Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | 42 | Custom Engine | 150 | 0.343350 | 0.267644 | 81.33% | 90.00% | 8.855 s |
| 1 | 42 | PyTorch | 150 | 0.345774 | 0.266752 | 81.33% | 90.00% | 0.034 s |
| 2 | 1337 | Custom Engine | 150 | 0.342519 | 0.277316 | 82.00% | 90.00% | 9.549 s |
| 2 | 1337 | PyTorch | 150 | 0.344327 | 0.268845 | 81.33% | 90.00% | 0.018 s |
| 3 | 2024 | Custom Engine | 150 | 0.342424 | 0.266294 | 81.33% | 90.00% | 9.538 s |
| 3 | 2024 | PyTorch | 150 | 0.346088 | 0.277704 | 82.00% | 90.00% | 0.018 s |
| 4 | 7 | Custom Engine | 150 | 0.346857 | 0.264440 | 81.33% | 90.00% | 9.486 s |
| 4 | 7 | PyTorch | 150 | 0.340047 | 0.268976 | 82.00% | 90.00% | 0.018 s |
| 5 | 999 | Custom Engine | 150 | 0.342613 | 0.263271 | 81.33% | 90.00% | 9.483 s |
| 5 | 999 | PyTorch | 150 | 0.345015 | 0.268710 | 81.33% | 90.00% | 0.018 s |
| **MEDIAN** | **All (5 seeds)** | **Custom Engine** | **150** | **0.342613** | **0.266294** | **81.33%** | **90.00%** | **9.486 s** |
| **MEDIAN** | **All (5 seeds)** | **PyTorch** | **150** | **0.345015** | **0.268845** | **81.33%** | **90.00%** | **0.018 s** |

### 4.3 Analysis of Training Convergence & Runtime Disparity

1. **Optimization Equivalence:** The median final training loss was `0.3426` for the custom engine and `0.3450` for PyTorch ($\Delta = 0.0024$). The median test loss was `0.2663` vs `0.2688` ($\Delta = 0.0026$). Both engines achieved identical median training accuracy (`81.33%`) and test accuracy (`90.00%`).
2. **Computational Overhead Analysis ($531.1\times$ latency ratio):**
   - The Custom Engine required a median of $9.486\text{ s}$ across 150 epochs, whereas PyTorch completed in $0.018\text{ s}$ ($17.86\text{ ms}$).
   - In the Custom Engine, each epoch instantiates $\sim 10,000$ Python objects (`Value`), dynamically builds set structures (`_prev`), and evaluates interpreted Python function closures (`_backward`) with pointer chasing across heap memory.
   - In PyTorch, the forward and backward passes execute via pre-compiled C++ ATen kernels utilizing contiguous CPU memory blocks and BLAS vectorization (`gemm`/`gemv`), eliminating Python runtime interpreter dispatch.

---

## 5. Task 6: PyTorch Framework Performance & Memory Layouts

### 5.1 Computation Graph Inspection

Inspecting the PyTorch backward DAG for the $2 \rightarrow 8 \rightarrow 1$ network reveals the internal operator chain constructed by PyTorch's autograd engine:

```text
├── SumBackward0
  ├── SigmoidBackward0
    ├── AddmmBackward0
      ├── AccumulateGrad (fc2.bias)
      ├── SigmoidBackward0
        ├── AddmmBackward0
          ├── AccumulateGrad (fc1.bias)
          ├── AccumulateGrad (input x)
          ├── TBackward0
            ├── AccumulateGrad (fc1.weight)
      ├── TBackward0
        ├── AccumulateGrad (fc2.weight)
```

The TorchScript intermediate representation (IR) captures the compiled execution graph:
```python
def forward(self, x: Tensor) -> Tensor:
    fc2 = self.fc2
    fc1 = self.fc1
    input = torch.sigmoid((fc1).forward(x, ))
    return torch.sigmoid((fc2).forward(input, ))
```

### 5.2 Eager vs Traced vs Compiled Benchmarking (from `execution_benchmarks.csv`)

Inference latency was benchmarked across 1,000 timed iterations preceded by 100 warmup iterations with batch size 32:

| Execution Mode | Status | Median Latency (µs) | Mean Latency (µs) | Throughput (Inf/sec) | Speedup vs Eager | Notes |
|---|---|---|---|---|---|---|
| **Eager** | SUCCESS | **`9.96 µs`** | 10.13 µs | 3,159,222 | $1.00\times$ | Standard PyTorch execution with C++ ATen dispatch. |
| **Traced (`torch.jit.trace`)** | SUCCESS | **`7.13 µs`** | 7.34 µs | 4,357,748 | **`1.40x`** | Bypasses Python dispatch; executes optimized TorchScript C++ runtime graph. |
| **Scripted (`torch.jit.script`)** | SUCCESS | **`6.12 µs`** | 6.19 µs | 5,172,434 | **`1.63x`** | Fully compiled TorchScript AST representation; eliminates interpreter overhead. |
| **Compiled (`aot_eager`)** | SUCCESS | **`77.46 µs`** | 78.72 µs | 406,493 | $0.13\times$ | PyTorch Dynamo frame guard inspection overhead dominates for tiny micro-models. |
| **Compiled (`inductor`)** | UNSUPPORTED | N/A | N/A | N/A | N/A | Raised `BackendCompilerFailed: backend='inductor' raised: CppCompileError: C++ compile error`. |

> [!IMPORTANT]
> **Compilation Overhead Analysis:** Graph compilation via `torch.compile` is not universally faster for small workloads. For tiny networks ($2 \rightarrow 8 \rightarrow 1$) with small batch sizes ($N=32$), Dynamo frame analysis and guarding introduce a fixed overhead of $\sim 65\text{--}75\ \mu\text{s}$, making `aot_eager` roughly $7.8\times$ slower than standard eager mode. In contrast, TorchScript tracing (`torch.jit.trace`) eliminates Python dispatch without introducing Dynamo guard overhead, yielding a genuine **$1.40\times$ speedup**, while `torch.jit.script` achieves a **$1.63\times$ speedup**.
>
> Regarding `torch.compile(backend="inductor")`: On this local macOS ARM64 environment, invoking Inductor triggered an exception of type `BackendCompilerFailed` during the C++ compilation phase (`CppCompileError`). As required by the lab instructions, this failure is recorded transparently with its exact exception type and message, and TorchScript (`torch.jit.trace`/`torch.jit.script`) is evaluated as the fully functional JIT compilation mechanism.

---

### 5.3 Tensor Layout & Contiguity Analysis (from `tensor_layout_benchmarks.csv`)

To investigate the micro-architectural consequences of memory layout on hardware caches, a $2048 \times 2048$ single-precision tensor ($16\text{ MB}$) was benchmarked across contiguous and non-contiguous memory layouts:
1. **Contiguous Tensor ($X$):** Stored in row-major order with shape `[2048, 2048]` and stride `[2048, 1]`.
2. **Non-Contiguous Transposed View ($Y = X^T$):** Shares the exact same underlying storage buffer, but strides are inverted to `[1, 2048]`.

#### Tensor Metadata Inspection:

| Property | Contiguous Tensor (`x_contig`) | Transposed View (`x_noncontig`) |
|---|---|---|
| **Shape** | `[2048, 2048]` | `[2048, 2048]` |
| **Dtype** | `torch.float32` | `torch.float32` |
| **Stride** | `[2048, 1]` | `[1, 2048]` |
| **Contiguity (`is_contiguous`)** | `True` | `False` |
| **Element Size (bytes)** | `4` | `4` |
| **Total Elements** | `4,194,304` | `4,194,304` |
| **Total Memory (KB)** | `16,384.0 KB` | `16,384.0 KB` |

#### Empirical Benchmark Results (500 Timed Iterations)

| Operation | Contiguous Stride | Non-Contiguous Stride | Contiguous Median (ms) | Non-Contiguous Median (ms) | Slowdown Factor | Contiguous Bandwidth | Non-Contiguous Bandwidth |
|---|---|---|---|---|---|---|---|
| **Row-wise Reduction (`sum(dim=1)`)** | `[2048, 1]` | `[1, 2048]` | **`0.182 ms`** | **`0.495 ms`** | **`2.73x slower`** | **`92.37 GB/s`** | **`33.89 GB/s`** |
| **Column-wise Reduction (`sum(dim=0)`)** | `[2048, 1]` | `[1, 2048]` | **`0.400 ms`** | **`0.173 ms`** | **`0.43x (faster)`** | **`41.92 GB/s`** | **`97.00 GB/s`** |
| **Elementwise Addition (`t + 1.0`)** | `[2048, 1]` | `[1, 2048]` | **`0.455 ms`** | **`0.439 ms`** | **`0.96x`** | **`36.86 GB/s`** | **`38.24 GB/s`** |
| **Elementwise Mult (`t * 2.0`)** | `[2048, 1]` | `[1, 2048]` | **`0.423 ms`** | **`0.425 ms`** | **`1.01x`** | **`39.68 GB/s`** | **`39.48 GB/s`** |

#### Hardware Micro-Architectural Explanation:
1. **Cache Spatial Locality:** A standard CPU cache line is 64 bytes (16 `float32` values). In a contiguous row-wise reduction (`sum(dim=1)`), reading element $x_{i, j}$ triggers the CPU hardware prefetcher to load the subsequent 15 floats into the L1 data cache, yielding a near-100% L1 cache hit rate and reaching $92.37\text{ GB/s}$ effective bandwidth.
2. **Strided Memory Access & Cache Thrashing:** In the transposed view (`stride=[1, 2048]`), advancing along dimension 1 requires jumping 8,192 bytes in memory for every single float. Only 4 bytes out of each 64-byte fetched cache line are consumed before eviction, causing severe cache thrashing and dropping effective memory bandwidth by $63.3\%$ to $33.89\text{ GB/s}$ ($2.73\times$ slowdown).
3. **Symmetry Inversion on Column Reduction:** When performing `sum(dim=0)`, the reduction moves along the columns. For the contiguous tensor, column traversal jumps 2,048 elements ($0.400\text{ ms}$), whereas for the transposed tensor, column traversal is contiguous along physical memory ($0.173\text{ ms}$), perfectly confirming the memory access model.
4. **Reshape Constraints:** Attempting to call `.view(-1)` on the transposed non-contiguous tensor raises `RuntimeError: view size is not compatible with input tensor's size and stride`, necessitating an explicit `.contiguous()` call which forces a memory allocation and byte-copy.

---

## 6. Limitations & Technical Boundaries

1. **Scalar Granularity Memory Overhead:** The custom engine operates on individual scalar `Value` nodes rather than multi-dimensional tensor buffers. For large datasets or architectures (e.g. ResNet), the graph would require millions of heap-allocated Python objects, exceeding system memory and interpreter capability.
2. **Absence of Vectorized SIMD Kernels:** The custom engine evaluates mathematical expressions through interpreted Python operators rather than ARM Neon or AVX-512 SIMD vector registers, accounting for the $\approx 531\times$ training latency gap compared to PyTorch.
3. **Platform-Specific `torch.compile` Inductor Limitation:** In this local environment, PyTorch 2.3.1's Inductor backend raises `BackendCompilerFailed: backend='inductor' raised: CppCompileError: C++ compile error` when compiling C++ artifacts. TorchScript tracing (`torch.jit.trace`) and scripting (`torch.jit.script`) serve as the robust, production-grade compiled graph execution mechanisms.

---

## 7. Conclusions

1. **Mathematical Robustness:** The reverse-mode automatic differentiation engine was proven mathematically correct: gradients matched analytical values and PyTorch autograd to $0.000$ error, and numerical finite differences to $< 7.43 \times 10^{-9}$ error, well exceeding the required $< 1.0 \times 10^{-6}$ precision threshold.
2. **Genuine Training Convergence:** The custom engine successfully trained a $2 \rightarrow 8 \rightarrow 1$ neural network on `make_moons`, demonstrating statistical and numerical convergence parity with PyTorch across 5 reproducible seeds.
3. **Empirical Performance Insights:** Real-world benchmarking established that JIT compilation and memory contiguity have substantial performance impacts. TorchScript tracing provided a $1.40\times$ latency reduction for small-batch inference, while contiguous memory layouts accelerated reduction throughput by $2.73\times$ through hardware cache line spatial locality.
