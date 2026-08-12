<!-- SPDX-License-Identifier: Apache-2.0 -->
# PACI Benchmark Specification & Methodology (Schema v2)

This document defines the benchmarking methodology, execution protocols, and **Schema v2 JSON specification** for the PACI 3-tier adaptive cascade system.

---

## 1. Measured Benchmarking Philosophy

PACI enforces **Measured Honesty**:
- Every timing, latency, footprint, and instruction count figure reported in documentation must derive directly from executed benchmark artifacts (`outputs/bench/*.json`).
- Synthetic estimates, rounded-up numbers, or unverified hardware claims are explicitly rejected and caught by `tools/check_no_fixture_in_results.py`.

---

## 2. Benchmark Execution Units

The single measurement binary [`bench/bench_main.c`](../bench/bench_main.c) evaluates four primary units:

1. **`tier0_ekf`**: Measures hot-cache execution latency of the scalar Extended Kalman Filter and statistical gating logic (`paci_step` with NIS computation).
2. **`tier1_int4`**: Measures hot and cold (64 MB cache-thrashing) execution latency of the 4-bit INT4 Conv1D screening classifier (`paci_infer_t1_s4`).
3. **`tier2_int8`**: Measures hot and cold (64 MB cache-thrashing) execution latency of the 8-bit INT8 Conv1D fault classifier (`paci_infer_t2_s8`).
4. **`cascade_trace`**: Evaluates end-to-end trace execution across the standard 2,000-step process trace (`bench_trace.h`), measuring total execution time, $N_1$ Tier 1 invocations, $N_2$ Tier 2 invocations, and realized amortized per-step cost.

---

## 3. Schema v2 JSON Specification

Benchmark results are emitted in Schema v2 format:

```json
{
  "schema_version": 2,
  "target": "x86_64-native",
  "cpu": "Generic CPU",
  "compiler": "gcc 13.2.0",
  "flags": "-O2 -fno-fast-math -ffp-contract=off",
  "metric": "ns_median",
  "note": "Hot and cold cache latency in ns. Cold numbers use 64MB thrashing.",
  "batches": 31,
  "model_artifacts": {
    "tier1": "tier1_model.tflite",
    "tier2": "tier2_model.tflite"
  },
  "units": {
    "tier0_ekf": {
      "hot": {"value": 23982.12, "mad": 880.19}
    },
    "tier1_int4": {
      "hot": {"value": 23641.80, "mad": 557.84},
      "cold": {"value": 38200.00, "mad": 3050.00},
      "scratch_bytes": 8192
    },
    "tier2_int8": {
      "hot": {"value": 108390.67, "mad": 2977.01},
      "cold": {"value": 120590.00, "mad": 4940.00},
      "scratch_bytes": 8192
    },
    "cascade_trace": {
      "total": 10600000,
      "n1": 303,
      "n2": 11,
      "steps": 2000
    }
  }
}
```

---

## 4. Quality Flags & Statistical Metrics

- **Median & Median Absolute Deviation (MAD)**: Latency is reported as median $\pm$ MAD across 31 measurement batches (after discarding 3 initial warmup batches).
- **Variance Quality Flag**: If relative dispersion ($\text{MAD} / \text{median}$) exceeds $10\%$, `bench_main` logs an execution quality flag warning of high system jitter.
- **Volatile Sink**: All inference outputs are accumulated into a global `volatile float g_volatile_sink` to guarantee that compiler dead-code elimination does not prune benchmark loops.

---

## 5. Footprint Differencing Methodology (GATE 3.2)

Static binary section sizes are measured across four build variants compiled with `-ffunction-sections -fdata-sections` and linked with `-Wl,--gc-sections`:
- `core_only`: Flash/RAM baseline of physics, EKF, ring buffer, and cascade dispatch.
- `core+tier1`: Incremental footprint adding Tier 1 INT4 inference engine.
- `core+tier2`: Incremental footprint adding Tier 2 INT8 inference engine.
- `full`: Complete binary including measurement harness.

Per-tier Flash and RAM deltas are calculated via:
$$\Delta\text{Flash}_{\text{Tier 1}} = \text{Flash}(\text{core+tier1}) - \text{Flash}(\text{core\_only})$$
$$\Delta\text{RAM}_{\text{Tier 1}} = \text{RAM}(\text{core+tier1}) - \text{RAM}(\text{core\_only})$$

Results are written to `outputs/bench/footprint.json`.
