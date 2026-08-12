<!-- SPDX-License-Identifier: Apache-2.0 -->
# GATE 3.5 — Corstone-300 FVP Instruction Profiling Guide

This document provides step-by-step instructions for running exact instruction count profiling of the PACI 3-tier cascade on the **Arm Corstone-300 Fixed Virtual Platform (FVP)** featuring **Cortex-M55** with **Helium Vector Extension (MVE)**.

---

## 1. Prerequisites & Toolchain Setup

### A. GNU Arm Embedded Toolchain (`arm-none-eabi-gcc`)
1. Download Arm GNU Toolchain 13.x or later from [developer.arm.com](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads).
2. Add the toolchain `bin/` directory to your system `PATH`:
   ```bash
   export PATH="/path/to/arm-gnu-toolchain/bin:$PATH"
   ```
3. Verify installation:
   ```bash
   arm-none-eabi-gcc --version
   ```

### B. Arm Corstone-300 Ecosystem FVP
1. Download **Ecosystem FVP for Corstone-300** from [Arm Ecosystem FVPs](https://developer.arm.com/downloads/-/arm-ecosystem-fvps).
2. Unpack and add the executable `FVP_Corstone_SSE-300_Ethos-U55` to your `PATH`:
   ```bash
   export PATH="/path/to/FVP_Corstone-300/bin:$PATH"
   ```
3. Verify installation:
   ```bash
   FVP_Corstone_SSE-300_Ethos-U55 --version
   ```

---

## 2. Running Automated Instruction Profiling

The harness [`bench/fvp_profile.py`](../bench/fvp_profile.py) automates cross-compilation, FVP invocation, trace log collection, two-point differencing, and Schema v2 JSON output.

### Command Execution
```bash
python bench/fvp_profile.py --output outputs/bench/cortex-m55-fvp.json
```

### Dry-Run Verification (Prerequisite Check)
To check whether the cross-compiler and FVP binary are present in your current environment without starting a run:
```bash
python bench/fvp_profile.py --dry-run
```

---

## 3. Manual Build & Invocation Protocol

If you prefer to manually compile and run the FVP model:

### Step 1: Cross-Compile Binary
```bash
cmake -S . -B build_m55 \
  -DCMAKE_C_COMPILER=arm-none-eabi-gcc \
  -DCMAKE_SYSTEM_NAME=Generic \
  -DCMAKE_SYSTEM_PROCESSOR=arm \
  -DCMAKE_C_FLAGS="-mcpu=cortex-m55 -mfloat-abi=hard -mfpu=fpv5-d16 -O2 -fno-fast-math -ffp-contract=off" \
  -DPACI_BUILD_BENCH=ON

cmake --build build_m55 --target bench_main
```

### Step 2: Run FVP with Instruction Statistics
```bash
FVP_Corstone_SSE-300_Ethos-U55 \
  -a build_m55/bench/bench_main.elf \
  -C core_clk.params=0 \
  --stat \
  -o outputs/bench/fvp_trace.log
```

---

## 4. Measuring Instruction Differencing

Instruction count differencing measures exact executed instructions between execution unit boundaries:
- **Tier 0 EKF**: $\Delta I_{\text{EKF}} = I_{\text{post\_ekf}} - I_{\text{pre\_ekf}}$
- **Tier 1 INT4**: $\Delta I_{\text{T1}} = I_{\text{post\_t1}} - I_{\text{pre\_t1}}$
- **Tier 2 INT8**: $\Delta I_{\text{T2}} = I_{\text{post\_t2}} - I_{\text{pre\_t2}}$
- **Cascade Trace**: Total instructions across 2,000 steps divided by 2,000 to obtain amortized instructions per step.

The results are parsed and formatted into `outputs/bench/cortex-m55-fvp.json` conforming strictly to **Schema v2**.
