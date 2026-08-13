# Measurement Methodology

**Project**: PACI (Physics-Informed Anomaly Classification for TinyML)  
**Target**: Arm AI Optimization Challenge 2026 — Phase 6 Deliverable  

---

This document outlines the source of truth for the performance claims presented in our submission, explicitly distinguishing between metrics that are directly measured on hardware/FVP, derived from code execution traces, or modelled theoretically.

## 1. Measured Metrics

These numbers are obtained via physical observation or cycle-accurate simulation using official Arm Corstone-300 FVP or native x86/aarch64 profiling.

- **Execution Latency (Cycle Count)**
  - Measured via ARM DWT CYCCNT registers on Cortex-M / Corstone-300 FVP.
  - Measured via `CLOCK_MONOTONIC_RAW` (`bench_timing.c`) for native targets (aarch64/x86_64).
  - Subject to variance on native host due to OS scheduler jitter; reported as Median ± MAD (Median Absolute Deviation).
- **Binary Footprint (Code/Data/BSS)**
  - Measured via `size` on stripped elf binaries compiled with `-O2` and the specific PACI toolchain variations (`core_only`, `core+tier1`, `core+tier2`, `full`).
- **CMSIS-NN Requantization Bias**
  - Measured empirically via `PACI_TRACE_REQUANT` macro intercepting scalar values immediately pre- and post-rounding during live data streams, and verified dynamically.

## 2. Derived Metrics

These metrics are calculated unambiguously from logged deterministic state during the execution pipeline.

- **Cascade Wake Ratios (N1 / N2 triggers)**
  - Derived by counting NIS threshold breaches and EKF stability failures over the evaluation test suite (`tests/test_infer_t2.py`).
  - Completely deterministic based on the provided physics simulation data and specific EKF covariance updates.
- **Dynamic Energy Profile (Effective Average Cost)**
  - Derived formula: $E_{\text{avg}} = (N_{T0} E_{T0} + N_{T1} E_{T1} + N_{T2} E_{T2}) / N_{\text{total}}$
  - Computed explicitly by our Python benchmarking harness (`bench/report.py`) reading the JSON trace dumped by `bench_main.exe --cascade`.

## 3. Modelled Metrics

These metrics estimate real-world analog phenomenon or generalize physical bounds without a physical IC implementation in hand.

- **Rounding Bias Worst-Case Bound (1.25 LSBs)**
  - Modelled mathematically based on maximum possible uniform quantization error across a 3-layer cascade assuming perfectly worst-case correlation.
- **Power Consumption estimates (if presented)**
  - Modelled using generic Cortex-M55 dynamic power figures (uW/MHz) extrapolated onto our measured cycle counts. (Note: Only cycle counts are strictly submitted for judging, power extrapolations are illustrative only).
- **Sensor Input Sampling Period**
  - Modelled based on IMU typical output data rates (100 Hz / 10ms).

---
*No results are fabricated. If hardware was unavailable, numbers are restricted strictly to FVP / aarch64 native outputs as mandated by the challenge rules.*
