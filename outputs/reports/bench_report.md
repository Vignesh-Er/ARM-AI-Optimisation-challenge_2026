# PACI Benchmark Report: `x86_64-native`

> Automated hardware benchmark evaluation conforming to **Schema v3** on **x86_64-native**.

## 1. Execution Latency

Measured across **31 batches**. Initialization and setup are excluded from per-sample measurements.

| Tier / Execution Unit | Hot Latency (Median ± MAD) | Cold Latency (Median ± MAD) | Scratch Buffer |
|:---|:---:|:---:|:---:|
| **Tier 0 (EKF)** | `465.20 ± 12.22 ns` | — | N/A (0 B) |
| **Tier 1 (INT4)** | `22422.87 ± 535.17 ns` | `27970.00 ± 1240.00 ns` | 8,192 B (8.00 KB) |
| **Tier 2 (INT8)** | `110517.87 ± 1324.04 ns` | `130790.00 ± 6620.00 ns` | 8,192 B (8.00 KB) |

### Statistical Detail
| Component | Mean | Min | Max | Samples |
|:---|:---:|:---:|:---:|:---:|
| Tier 0 Hot | 463.82 | 428.57 | 514.32 | 31 |
| Tier 1 Hot | 22548.86 | 21262.81 | 24887.45 | 31 |
| Tier 1 Cold | 31171.61 | 25260.00 | 113320.00 | 31 |
| Tier 2 Hot | 110854.26 | 103774.68 | 118287.45 | 31 |
| Tier 2 Cold | 130303.55 | 112230.00 | 179810.00 | 31 |

## 2. Cascade Trace Results

End-to-end evaluation across standard **2,000 steps** input trace (`k=1.0` benchmark sequence).

| Cascade Metric | Realized Value | Performance Analysis |
|:---|:---:|:---|
| **Total Evaluation Time** | `23.020 ms (23,020,300.0 ns)` | Cumulative trace execution time |
| **Total Steps Evaluated** | `2,000` | Standard deterministic benchmark length |
| **Tier 1 Invocations ($N_1$)** | `303` | Escalation rate: **15.15%** (84.85% filtered by Tier 0) |
| **Tier 2 Invocations ($N_2$)** | `90` | Escalation rate: **4.50%** (70.30% resolved by Tier 1) |
| **Effective Per-Step Latency** | `11510.15 ns/step` | Realized amortized execution cost per sample |
| **Always-On Tier 2 Baseline** | `122209.20 ns/step` | Un-gated Tier 2 execution on every step |
| **Compute Latency Reduction** | **90.58%** | Relative savings vs Always-On Tier 2 baseline |
| **Effective Speedup Factor** | **10.62×** | Realized acceleration from adaptive tri-tier gating |

> **Note:** The PACI cascade reduces computational work by 90.6%, providing a strong proxy for reduced dynamic compute energy.

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

