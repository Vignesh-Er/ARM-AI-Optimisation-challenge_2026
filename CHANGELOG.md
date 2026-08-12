# CHANGELOG

All notable changes and GATE milestones for the PACI-Arm project are documented in this file.

---

## [Phase 3] - 2026-08-12 — Measurement Harness & CI Pipeline

### Added
- **GATE 3.1 (`bench/bench_main.c`)**: Standalone single measurement binary evaluating `tier0_ekf`, `tier1_int4` (hot/cold), `tier2_int8` (hot/cold), and 2,000-step static `cascade_trace`. Generates Schema v2 JSON (`outputs/bench/native-smoke.json`).
- **GATE 3.2 (`bench/CMakeLists.txt`)**: Footprint differencing targets (`core_only`, `core+tier1`, `core+tier2`, `full`) linked with `-Wl,--gc-sections`. Recorded to `outputs/bench/footprint.json`.
- **GATE 3.3 (`bench/report.py`)**: Automated Schema v2 report generator producing Markdown summary tables and latency breakdown plots.
- **GATE 3.4 (`.github/workflows/arm-bench.yml`)**: CI workflow targeting `ubuntu-24.04-arm` runners with multi-target compilation, pytest verification, fixture check, and step summary publication.
- **GATE 3.5 (`bench/fvp_profile.py` & `docs/FVP_INSTRUCTIONS.md`)**: Corstone-300 Cortex-M55 + Helium instruction count differencing harness and guide.

---

## [Phase 2] - 2026-08-11 — CMSIS-NN Integration & Model Export

### Added
- **GATE 2.0**: Frozen window size to `PACI_WINDOW_SIZE=64`. Vendored CMSIS-NN v7.0.0 and CMSIS-DSP v1.17.1 as git submodules under `third_party/`.
- **GATE 2.1**: Trained Stage-A fixture models (`tier1_fixture.tflite`, `tier2_fixture.tflite`) and verified BatchNorm folding (`tools/verify_bn_fold.py`).
- **GATE 2.3/2.4**: Exported CMSIS-NN C weight headers (`tier1_weights.h`, `tier2_weights.h`). Implemented bit-exact C inference for Tier-2 INT8 (`paci_infer_t2_s8`) matching TFLite interpreter.
- **GATE 2.5**: Implemented bit-exact C inference for Tier-1 INT4 (`paci_infer_t1_s4`) matching NumPy reference. Generated requantization manifest (`outputs/models/requant_sites.json`).

---

## [Phase 1] - 2026-08-11 — Unified C Core Implementation

### Added
- Created production `paci_core/` C library:
  - `paci_physics.c`: Deal-Grove plasma etch rate calculation.
  - `paci_ekf.c`: Scalar EKF predict/update with reset guard.
  - `paci_ring.c`: Bounded, chronologically linearizing ring buffer.
  - `paci_cascade.c`: Multi-tier cascade dispatch and statistical gating.
- Replaced buggy legacy `Core/` skeleton (remediating defects D2, D3, D4, D5).
- Created bit-exact verification test suite against EKF golden trace (`tests/test_bitexact.py`).

---

## [Phase 0] - 2026-08-10 — Compliance & Hygiene Scaffold

### Added
- Added Apache-2.0 `LICENSE` (remediating defect D1).
- Set up repository structure, `config.py`, and project hygiene.
