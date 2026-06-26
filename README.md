# CppNativeML - MNIST

A from-scratch C++ neural network library for MNIST digit classification.
Originally written as a DGIST CSE302 assignment; refactored, bug-fixed, and optimised for **Apple M3 Pro 36 GB** (arm64, macOS 26.5, Xcode / clang GNU++20).

---

## Final result

| Dataset | Accuracy |
|---|---|
| Train (60,000) | **99.91%** (59,943 / 60,000) |
| Test (10,000) | **98.33%** (9,833 / 10,000) |

Per-digit test breakdown:

| Digit | Correct | Total | Accuracy |
|---|---|---|---|
| 0 | 971 | 980 | 99.08% |
| 1 | 1124 | 1135 | 99.03% |
| 2 | 1017 | 1032 | 98.55% |
| 3 | 992 | 1010 | 98.22% |
| 4 | 964 | 982 | 98.17% |
| 5 | 872 | 892 | 97.76% |
| 6 | 942 | 958 | 98.33% |
| 7 | 1009 | 1028 | 98.15% |
| 8 | 954 | 974 | 97.95% |
| 9 | 988 | 1009 | 97.92% |

---

## Architecture

$$\text{Input}\ (784) \;\longrightarrow\; \text{Dense}\ 256,\ \text{ReLU} \;\longrightarrow\; \text{Dense}\ 10,\ \text{Softmax} \;\longrightarrow\; \text{Output}\ (10)$$

| Layer | Weights | Init |
|---|---|---|
| Hidden | $784 \times 256$ | He uniform $\left(\pm\sqrt{2/784} \approx \pm 0.0505\right)$ |
| Output | $256 \times 10$ | Xavier uniform $\left(\pm\sqrt{1/256} \approx \pm 0.0625\right)$ |

**Loss**: categorical cross-entropy
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n} \log h_i[\text{correct class}]$$

**Regularization**: L2 weight decay $(\lambda = 10^{-4})$

---

## Training

Mini-batch SGD with a 3-phase learning-rate schedule.

| Phase | Epochs | Steps | $\theta$ |
|---|---|---|---|
| 1 — rapid descent | 5 | 4,685 | 0.30 |
| 2 — plateau | 8 | 7,496 | 0.20 |
| 3 — fine-tuning | 7 | 6,559 | 0.10 |
| **Total** | **20** | **18,740** | — |

Batch size: 64 samples ($\approx 937$ steps / epoch). Indices are shuffled once per epoch.

$$\text{Initial CE: } 2.3069 \quad\longrightarrow\quad \text{Final CE (full train set): } 0.0114$$

---

## Platform

| Item | Value |
|---|---|
| CPU | Apple M3 Pro (6P + 6E cores, arm64) |
| RAM | 36 GB unified memory |
| Compiler | Apple clang (Xcode 26), `-std=gnu++20` |
| SIMD | ARM NEON (`float64x2_t`) via `<arm_neon.h>` |

All SIMD paths are guarded with `#if defined(...)` so the code compiles on x86-64 (AVX2/FMA, SSE2) and generic platforms without modification.

---

## Directory layout

```
CppNativeML/
├── main.cpp
├── linAlge/
│   └── matrix.h                     mat class -- flat row-major, SIMD, Z-order Winograd
├── optimizer/
│   └── GradientDescent.h            mini-batch SGD with backprop
├── model/
│   ├── actFunction/
│   │   ├── logisticAct.h            sigmoid / dlogistic
│   │   └── reluAct.h                relu / drelu
│   └── objFunction/
│       ├── CrossEntropy.h           categorical CE + softmax, binary CE, dCE
│       └── MSE.h                    MSE loss + gradient
├── MNIST/
│   ├── readMNIST.h                  IDX file reader
│   └── archive/                     raw MNIST binary data (not tracked)
└── bmp/
    ├── bmp_writer.h
    ├── bmplot.h
    └── writeNumber.h
```

Header search path (Xcode target setting): `$(SRCROOT)/CppNativeML`

---

## Build

```bash
xcodebuild -project CppNativeML.xcodeproj \
           -scheme CppNativeML \
           -configuration Release \
           build
```

MNIST paths are resolved relative to `__FILE__` at compile time, so the binary works regardless of the Xcode scheme's working directory setting.

---

## Key optimisations

### `linAlge/matrix.h`

#### Flat row-major storage

Single `std::vector<double>`; element $(i,j)$ at index $i \cdot m + j$.
`row(i)` returns `data + i*m` — a single pointer used by every SIMD kernel, no per-row indirection.

#### SIMD — `axpy` inner kernel

Hot path of matrix multiply: $\text{ret}[i][j] \mathrel{+}= A[i][k] \cdot B[k][j]$ across all $j$.

| Macro | ISA | Width |
|---|---|---|
| `MAT_AVX2_FMA` | x86 AVX2+FMA | $4 \times$ `double` / iter (fused) |
| `MAT_AVX2` | x86 AVX2 | $4 \times$ `double` / iter |
| `MAT_SSE2` | x86 SSE2 | $2 \times$ `double` / iter |
| `MAT_NEON` | ARM NEON (M3 Pro) | $2 \times$ `double` / iter via `vmlaq_f64` |
| *(fallback)* | scalar | $1 \times$ `double` / iter |

#### Loop order and tiling — `nProduct`

Loop order **i-k-j** keeps $\text{ret}[i]$ hot in L1 across the entire $k$ loop and streams $B[k]$ sequentially.
Outer $i$ and $k$ loops are tiled at $T_i = T_k = 48$: three $48 \times 48$ `double` tiles occupy $3 \times 48^2 \times 8 = 55\ \text{KB}$, within the M3 Pro 128 KB L1 cache.
`nProduct` partitions the $i$ dimension across `hardware_concurrency()` threads; spawn is skipped below $2^{16}$ flops.

#### Z-order (Morton) layout for Winograd D\&C

`Winograd()` (Strassen-variant) packs both inputs into Z-order before recursing, then unpacks the result.
The Z-index of $(i,j)$ is defined by bit-interleaving:

$$\text{spread}_{16}(x): \text{bit } k \text{ of } x \;\mapsto\; \text{bit } 2k \text{ of result}$$
$$\text{zidx}(i,j) = \text{spread}_{16}(j) \;\big|\; \bigl(\text{spread}_{16}(i) \ll 1\bigr)$$

Any aligned $2^k \times 2^k$ submatrix occupies a contiguous $4^k$-element block, so the four $N \times N$ quadrants of a $2N \times 2N$ matrix sit at pointer offsets $\{0,\, N^2,\, 2N^2,\, 3N^2\}$ — no copying per recursive call.
Workspace ($5P^2$ doubles, proved by geometric series $15(P/2)^2 \cdot 4/3 = 5P^2$) is pre-allocated once and carved by pointer arithmetic at each level.

#### Cache-friendly transpose — `T()`

$8 \times 8$ block transpose; each block ($512$ bytes $= 8$ cache lines) is processed with hardware-prefetcher-friendly sequential reads.

---

### `optimizer/GradientDescent.h`

#### Backprop correctness

Two bugs fixed from the original:

| Bug | Original | Fixed |
|---|---|---|
| Output delta | `dactf` applied to `dloss(x,Y)` | applied to saved activation $a_\text{last}$ |
| $\delta$ propagation | $x = a^\top \delta$, then `dactf`$(dW)$ | $dW = a^\top \delta$; $x = \delta W^\top$; then `dactf`$(a_\text{prev})$ |

#### `combined_output_delta` flag

For softmax + categorical CE, the combined output gradient simplifies to:

$$\frac{\partial \mathcal{L}}{\partial z} = \frac{h - Y}{n}$$

The softmax Jacobian and CE denominator cancel exactly, so no additional `dactf` call is needed on the output layer.
The same formula holds for sigmoid + binary CE:

$$\frac{\partial \mathcal{L}_\text{BCE}}{\partial h} \odot h(1-h) = \frac{h - Y}{n}$$

Setting `combined_output_delta=true` skips the redundant multiplication.

#### Mini-batch SGD

`batch_size` parameter ($0$ = full batch). At each epoch boundary, indices are shuffled with `std::mt19937`; $b$ rows are extracted per step via `memcpy`. Batch buffers are pre-allocated outside the loop to avoid per-step heap traffic.

Weight update per step:

$$W \;\leftarrow\; W(1 - \theta\lambda) \;-\; \theta \cdot \frac{\partial \mathcal{L}}{\partial W}$$

#### SIMD weight update — `subScaledInPlace`

$W \mathrel{-}= \theta \cdot dW$ applied row-by-row:
`_mm256_fnmadd_pd` (AVX2+FMA) or `vsubq_f64(d, vmulq_f64(scale, s))` (NEON).

---

### Activation functions

| Function | Forward | Backward |
|---|---|---|
| Sigmoid | $\sigma(x) = \dfrac{1}{1+e^{-x}}$ | $\sigma'(a) = a(1-a)$ |
| ReLU | $\text{ReLU}(x) = \max(0,x)$ | $\text{ReLU}'(a) = \mathbf{1}[a > 0]$ |

ReLU is used for the hidden layer to avoid the $\leq 0.25\times$ gradient attenuation of sigmoid (since $\sigma'(z) \leq 0.25$ everywhere).
He init $\left(\pm\sqrt{2/\text{fan} \textunderscore \text{in}}\right)$ is matched to ReLU; Xavier $\left(\pm\sqrt{1/\text{fan} \textunderscore \text{in}}\right)$ is used for the softmax output layer.

#### Why sigmoid saturates and dies

With large $\theta$: if pre-activation magnitude grows, $\sigma(z) \to 0$ and $\sigma'(z) = a(1-a) \to 0$.
Once outputs saturate, every gradient is zero regardless of $\theta$ — the network cannot recover without re-initialisation.

#### Why `dMSE` must divide by $n$

Without $/n$:

$$\frac{\partial \text{MSE}}{\partial h} = 2(h - Y)$$

Effective learning rate becomes $\theta \cdot n$. With $n = 60{,}000$ and $\theta = 0.3$, effective lr $= 18{,}000$ — weights explode in step 1, sigmoid saturates, MSE $= 1.0$, all gradients $= 0$: stuck permanently.

---

## Non-standard operator conventions

`mat` intentionally uses two operators with non-standard semantics:

| Operator | Conventional meaning | Actual behaviour in `mat` |
|---|---|---|
| `a < b` | less-than comparison | **copies** `b` into `a`; returns `mat&` |
| `-a` | returns a negated copy | **negates `a` in place**; returns `mat&` |

---

## Bugs fixed from original source

Attention: The original code is currently inaccessible. The original source code was authored in 2023.

| File | Bug |
|---|---|
| `GradientDescent.h` | $\delta_0$: `dactf` applied to `dloss` output instead of activation $a_\text{last}$ |
| `GradientDescent.h` | Backward loop: `dactf` applied to weight gradient $dW$ instead of propagated $\delta$ |
| `MSE.h` | `dMSE` missing $/n$ — effective lr $= \theta \cdot n = 18{,}000$ causing weight explosion and dead network |
| `matrix.h` | `operator-()` iterated $j < n$ instead of $j < m$ (non-square matrices silently wrong) |
| `matrix.h` | Member `int i,j,k` shared across methods causing data race under multithreading |
| `matrix.h` | `zadd(C12, m6, C12, h)` aliasing UB with `__restrict__` — fixed via S1 intermediate |
| `matrix.h` | 14 `std::vector` allocations per `winograd_z` call ($\approx 94\text{M}$ heap ops for $N=512$); fixed with pre-allocated workspace |
| `model/regularization/L1L2.h` | Broken relative `#include "matrix.h"` path |
| All headers | Non-template free functions without `inline` causing ODR / duplicate-symbol link error |
| `bmp/bmp_writer.h` | Global arrays without `static` causing ODR violation |
| `optimizer/GA.h` | `std::random_device rd` at namespace scope without `inline` causing ODR violation |

---

## Before testing this Code
The MNIST dataset is not included in this repository. Please download the dataset from a reputable source, such as Kaggle. You can print the MNIST dataset as a BMP image using the BMP::bmp::writeBMP() function.
#### Example Code
```cpp
    BMP::bmp shader(28, 28);
    
    auto MNIST_train_4 = train[4].first;
    std::cout << "MNIST_train_4 Label: " << train[4].second << std::endl;
    
    for (int i=0; i<28; ++i) {
        for (int j=0; j<28; ++j) {
            shader.red(i,j) = shader.blue(i,j) = shader.green(i,j) = MNIST_train_4[i+j*28];
        }
    }
    
    shader.writeBMP("./", "MNIST_train_4");
```

#### Result of Example Code: Image and Label
**Printed Image**:
[MNIST_train_4.bmp](https://github.com/user-attachments/files/29320251/MNIST_train_4.bmp)

**Output**:
```
MNIST_train_4 Label: 9
```

