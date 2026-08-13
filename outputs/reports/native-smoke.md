# PACI Benchmark Report: `x86_64-native`

> Automated hardware benchmark evaluation conforming to **Schema v3** on **x86_64-native**.

## 1. Execution Latency

Measured across **31 batches**. Initialization and setup are excluded from per-sample measurements.

| Tier / Execution Unit | Hot Latency (Median ± MAD) | Cold Latency (Median ± MAD) | Scratch Buffer |
|:---|:---:|:---:|:---:|
| **Tier 0 (EKF)** | `456.41 ± 6.13 ns` | — | N/A (0 B) |
| **Tier 1 (INT4)** | `22592.64 ± 211.50 ns` | `34450.00 ± 2270.00 ns` | 8,192 B (8.00 KB) |
| **Tier 2 (INT8)** | `110620.26 ± 4589.43 ns` | `115630.00 ± 5350.00 ns` | 8,192 B (8.00 KB) |

### Statistical Detail
| Component | Mean | Min | Max | Samples |
|:---|:---:|:---:|:---:|:---:|
| Tier 0 Hot | 459.71 | 443.30 | 494.04 | 31 |
| Tier 1 Hot | 23917.67 | 22205.78 | 30795.02 | 31 |
| Tier 1 Cold | 33767.10 | 25590.00 | 44370.00 | 31 |
| Tier 2 Hot | 115974.05 | 104342.73 | 151930.18 | 31 |
| Tier 2 Cold | 118065.48 | 106610.00 | 135050.00 | 31 |

## 2. Cascade Trace Results

End-to-end evaluation across standard **2,000 steps** input trace (`k=1.0` benchmark sequence).

| Cascade Metric | Realized Value | Performance Analysis |
|:---|:---:|:---|
| **Total Evaluation Time** | `21.206 ms (21,205,500.0 ns)` | Cumulative trace execution time |
| **Total Steps Evaluated** | `2,000` | Standard deterministic benchmark length |
| **Tier 1 Invocations ($N_1$)** | `303` | Escalation rate: **15.15%** (84.85% filtered by Tier 0) |
| **Tier 2 Invocations ($N_2$)** | `90` | Escalation rate: **4.50%** (70.30% resolved by Tier 1) |
| **Effective Per-Step Latency** | `10602.75 ns/step` | Realized amortized execution cost per sample |
| **Always-On Tier 2 Baseline** | `105085.20 ns/step` | Un-gated Tier 2 execution on every step |
| **Compute Latency Reduction** | **89.91%** | Relative savings vs Always-On Tier 2 baseline |
| **Effective Speedup Factor** | **9.91×** | Realized acceleration from adaptive tri-tier gating |

> **Note:** The PACI cascade reduces computational work by 89.9%, providing a strong proxy for reduced dynamic compute energy.

## 3. Memory & Static Footprint

## 4. Platform & Build Metadata

| Configuration Property | Value |
|:---|:---|
| **Target Platform** | `x86_64-native` |
| **CPU / Core** | `Generic CPU` |
| **Compiler** | `gcc 13.2.0 ` |
| **Build Type** | `Unknown` |
| **Compiler Flags** | `-O2 -fno-fast-math -ffp-contract=off` |
| **Git Commit** | `unknown` |
| **Timestamp** | `` |
| **Primary Metric** | `ns_median` |
| **Measurement Batches** | `31` |
| **Schema Version** | `v3` |
| **Harness Notes** | Hot and cold cache latency in ns. Cold numbers use 64MB thrashing. |

