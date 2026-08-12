<!-- SPDX-License-Identifier: Apache-2.0 -->
# Contributing to PACI-Arm

Thank you for contributing to the **PACI (Physics-Informed Anomaly Classification for TinyML)** project for the **Arm AI Optimization Challenge 2026**.

---

## 1. Core Principles & Coding Standards

### A. Measured Honesty
- Every reported metric (latency, RAM/Flash footprint, instruction count) must derive from an executed benchmark output (`outputs/bench/*.json`).
- Never hardcode, estimate, or paraphrase benchmark results in documentation or pull requests.
- All code must pass `python tools/check_no_fixture_in_results.py` before submission.

### B. Compiler & Execution Flags
- All C code must compile strictly under `-O2 -fno-fast-math -ffp-contract=off`.
- `-fno-fast-math` and `-ffp-contract=off` are strictly required to preserve IEEE-754 compliance, prevent FMA contraction divergence, and maintain 1 ULP bit-exactness between host (aarch64/x86_64) and target (Cortex-M) builds.

### C. Zero Dynamic Memory Allocation
- No calls to `malloc`, `calloc`, `realloc`, or `free` are permitted in `paci_core/` or embedded C harnesses.
- All scratch buffers must be statically allocated (e.g. `static int8_t paci_scratch[8192] __attribute__((aligned(16)))`) and verified at runtime against layer buffer requirements (`PACI_E_BUFSIZE`).

---

## 2. Development & Pull Request Workflow

1. **Branch Discipline**:
   - Create feature or GATE branches off `main` (e.g., `phase3-bench-harness`).
   - Do not push directly to `main`.

2. **Testing Protocol**:
   - Run the full pytest test suite before committing:
     ```bash
     python -m pytest tests/ -v
     ```
   - Ensure all 95+ unit, integration, and bit-exactness tests pass cleanly.

3. **Two-Stage Model Plan**:
   - Stage A fixture models are named `*_fixture.tflite` (for shape/exporter validation only).
   - Stage B release models are named `*_release.tflite` (for final benchmark reports).
   - Fixture names must never appear in published results artifacts.

4. **License Header**:
   - Include `// SPDX-License-Identifier: Apache-2.0` at the top of all C/Python files.
