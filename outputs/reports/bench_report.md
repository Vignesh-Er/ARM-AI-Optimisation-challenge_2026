# PACI Benchmark Report: `x86_64-native`

> Automated hardware benchmark evaluation conforming to **Schema v2** on **x86_64-native**.

## 1. Execution Latency

Measured across **31 batches** using median and Median Absolute Deviation (MAD).

| Tier / Execution Unit | Algorithm / Hardware Kernel | Hot Latency (Median ± MAD) | Cold Latency (Median ± MAD) | Scratch Buffer |
|:---|:---|:---:|:---:|:---:|
| **Tier 0 (EKF)** | Physics-residual scalar EKF + statistical gating | `23982.12 ± 880.19 ns` | — | N/A (0 B) |
| **Tier 1 (INT4)** | CMSIS-NN 4-bit Conv1D classifier (`arm_convolve_s4`) | `23641.80 ± 557.85 ns` | `38200.00 ± 3050.00 ns` | 8,192 B (8.00 KB) |
| **Tier 2 (INT8)** | CMSIS-NN 8-bit Conv1D classifier (`arm_convolve_wrapper_s8`) | `120288.49 ± 10775.45 ns` | `122300.00 ± 8420.00 ns` | 8,192 B (8.00 KB) |

## 2. Cascade Trace Results

End-to-end evaluation across standard **2,000 steps** input trace (`k=1.0` benchmark sequence).

| Cascade Metric | Realized Value | Performance Analysis |
|:---|:---:|:---|
| **Total Evaluation Time** | `12.997 ms (12,997,200.0 ns)` | Cumulative trace execution time |
| **Total Steps Evaluated** | `2,000` | Standard deterministic benchmark length |
| **Tier 1 Invocations ($N_1$)** | `303` | Escalation rate: **15.15%** (84.85% filtered by Tier 0) |
| **Tier 2 Invocations ($N_2$)** | `11` | Escalation rate: **0.55%** (96.37% resolved by Tier 1) |
| **Effective Per-Step Latency** | `6498.60 ns/step` | Realized amortized execution cost per sample |
| **Theoretical Decomposed Cost** | `28225.44 ns/step` | $T_0 + \frac{N_1}{N} T_1 + \frac{N_2}{N} T_2$ model prediction |
| **Always-On Tier 2 Baseline** | `144270.61 ns/step` | Un-gated Tier 2 execution on every step |
| **Compute Latency Reduction** | **95.50%** | Relative savings vs Always-On Tier 2 baseline |
| **Effective Speedup Factor** | **22.20×** | Realized acceleration from adaptive tri-tier gating |

## 3. Memory & Static Footprint

### Scratch Buffer Allocations

| Tier Component | Scratch Memory Allocation | Storage Location | Reusability Semantics |
|:---|:---:|:---|:---|
| **Tier 0 (EKF)** | `0 B` | Stack | Stack-allocated scalar operations only |
| **Tier 1 (INT4)** | `8,192 B (8.00 KB)` | Static RAM Arena | Reusable activation arena (shared sequentially) |
| **Tier 2 (INT8)** | `8,192 B (8.00 KB)` | Static RAM Arena | Reusable activation arena (shared sequentially) |

> [!NOTE]
> Static binary section sizes (`.text`, `.data`, `.bss`) were not recorded in this benchmark artifact. Scratch buffer requirements are reported directly from execution unit metadata.

## 4. Platform & Build Metadata

| Configuration Property | Value | Description / Scope |
|:---|:---|:---|
| **Target Platform** | `x86_64-native` | Hardware benchmark deployment target |
| **CPU / Core** | `Generic CPU` | Host or target microarchitecture |
| **Compiler** | `gcc 13.2.0` | Toolchain used for C harness compilation |
| **Compiler Flags** | `-O2 -fno-fast-math -ffp-contract=off` | Optimization and IEEE-754 compliance flags |
| **Primary Metric** | `ns_median` | Timing metric reported by benchmark units |
| **Measurement Batches** | `31` | Statistical sample count for median / MAD |
| **Tier 1 Artifact** | `tier1_model.tflite` | INT4 model flatbuffer / weight source |
| **Tier 2 Artifact** | `tier2_model.tflite` | INT8 model flatbuffer / weight source |
| **Schema Version** | `v2` | PACI Benchmark JSON Schema specification |
| **Harness Notes** | Hot and cold cache latency in ns. Cold numbers use 64MB thrashing. FVP numbers require instruction differencing. | Operational notes and execution details |

