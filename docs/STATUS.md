<!-- SPDX-License-Identifier: Apache-2.0 -->
# PACI-Arm build status

Target: Arm AI Optimization Challenge 2026, Track 1 (Physical AI), phase-1
(software) submission round. Deadline 2026-08-14 16:00 PDT.

## Correction (2026-08-10)

An earlier version of this document claimed no prior PACI codebase existed.
That was wrong: it was written after searching only the local machine and an
empty placeholder GitHub repo, without checking for the actual project
repository the brief was written against. The real source is
`github.com/Karthikdebuger/PACI` (2 commits, fully populated: `phase1_physics/`
through `phase6_benchmark/`, `Core/`, `config.py`, `outputs/`). It has been
cloned into this working directory with full history preserved (`git log`
shows both original commits). D1-D8 below are now confirmed against that
actual source, not asserted against an empty directory.

Remotes:
- `origin` → `github.com/Vignesh-Er/ARM-AI-Optimisation-challenge_2026` (submission target, currently empty, push destination)
- `upstream` → `github.com/Karthikdebuger/PACI` (original source, read reference only)

## Hardware scope for this round

`AVAILABLE_HARDWARE = none`. This is the phase-1 (software) round. Measurement
targets are:
  (a) aarch64 Linux (host / `ubuntu-24.04-arm` CI runner)
  (b) Cortex-M55 + Helium on the free Corstone-300 FVP (functional model,
      not cycle-accurate — see G4)

No STM32 or other physical board numbers appear anywhere in this submission.
STM32 + ST-Link is the expected measurement target for a phase-2 (hardware)
round if this submission advances; that is out of scope here and appears in
the README only as noted future work, never as an estimate.

## Port vs. fresh-build split

- **Carried over from `Karthikdebuger/PACI`, as-is or lightly extended**: the
  Python layer — `config.py`, `phase1_physics/` (physics model + fault
  injector), `phase2_ekf/`, `phase3_scheduler/`, `phase4_tinyml/` (dataset,
  model, training), `phase6_benchmark/`, `live_demo.py`, `requirements.txt`.
  This code is functionally sound and already produces the labelled 5-class
  dataset; D6 and D7 are fixed in place within `phase6_benchmark/`, not
  rewritten from scratch.
- **Written fresh**: `paci_core/` (replaces the `Core/` skeleton — D2-D5 are
  design requirements met by construction, per the interface contract in
  section 8 of the brief), CMSIS-NN integration (`tools/export_cmsisnn.py`),
  the measurement harness (`bench/`), and CI (`.github/workflows/`). The old
  `Core/` directory is removed once `paci_core/` reaches parity, not kept
  alongside it.

## Phase checklist

- [x] Phase 0 — Compliance and hygiene
- [ ] Phase 1 — Unified C core
- [ ] Phase 2 — CMSIS-NN integration and export
- [ ] Phase 3 — Measurement harness
- [ ] Phase 4 — Honest benchmark rebuild
- [ ] Phase 5 — Rounding-bias budget
- [ ] Phase 6 — CI and docs

## Confirmed defect list (D1-D8), verified against actual source on 2026-08-10

| ID | Verified against | Finding |
|----|-------------------|---------|
| D1 | repo root | No `LICENSE` file. Confirmed. Fixed this commit (Apache-2.0). |
| D2 | `Core/Src/main_cascade.c:56,65` | Confirmed exactly as described: `sensor_buffer[buffer_index % WINDOW_SIZE] = ...` then `Run_CNN_Inference(sensor_buffer)` passed directly, unlinearised. |
| D3 | `Core/Src/main_cascade.c:15` | Confirmed: `uint8_t buffer_index = 0;` — wraps at 256. |
| D4 | `Core/Inc/ekf.h:9-11` vs `config.py:31,37` | Confirmed exactly: C has `TAU 0.1f`, `EKF_Q_VAR 2.0f`; Python has `TAU = 0.3`, `Q_VAR = 0.5`. |
| D5 | `Core/Src/main_cascade.c:18-29` | Confirmed: `Run_CNN_Inference` is a `printf` returning 0; `Compute_Physics_Prediction` returns the constant `NOMINAL_ETCH_RATE`. |
| D6 | `phase6_benchmark/run_benchmarks.py:109-115` | Confirmed: `physics_cost_per_step = 1.0`, `cnn_cost_per_invocation = 50.0`, both invented, feeding `energy_saving_pct`. |
| D7 | `outputs/reports/benchmark_report.md` vs `README.md` | Confirmed: report shows Moving Average at 95.9% reduction / 100% detection / 3.6% false wake vs. PACI's 84.1% / 100% / 4.6% — Moving Average dominates on every column. README's results table shows only PACI and Always-On CNN; the Moving Average row (and three other baselines) is omitted. Root cause confirmed in `phase1_physics/synthetic_data.py`: every injected fault is far outside the sensor noise floor (σ=2.0 nm/min) — Sensor Fault ~125σ, Gas Leak ~23σ, Equipment Drift ~6σ after one step (`DRIFT_RATE=0.05` compounding, hits 125σ by step 20), Unexpected Deviation ~3σ. All baselines trivially hit 100% detection except Variance Threshold, so the benchmark isn't discriminating gating quality. |
| D8 | `tests/` | Confirmed: contains only `__init__.py`. |

## Fault-severity ladder (fix for the D7 root cause)

Per project-owner guidance: the existing fault magnitudes make the benchmark
too easy to be informative. Phase 4 will keep the class names and label
integers (0=Normal, 1=Sensor Fault, 2=Gas Leak, 3=Equipment Drift,
4=Unexpected Deviation unchanged — matches the trained model and existing
plots) but add a difficulty ladder parameterised roughly at 0.5σ / 1σ / 2σ /
5σ per class, with the headline benchmark run at the hard end. Equipment
Drift specifically will be slowed to ~0.1-0.3%/step (vs. the current 5%/step,
which reaches a 1750 nm/min rate within two minutes) so it takes 50+ steps to
clear the noise floor — the regime where a physics-residual gate should beat
a moving average, because the MA adapts to slow drift and normalises it away
while the physics prediction does not.

## Assumptions

Running log of `// ASSUMPTION:` decision points, appended here as they occur.
Printed in full to console at the end of the run per section 9.

- ASSUMPTION: submission repo (`origin`) starts from a full clone of
  `Karthikdebuger/PACI` rather than a fresh empty history, so the commit trail
  (incl. the two pre-existing commits) is preserved end to end — chosen per
  project-owner instruction that a repo with real history reads better to a
  judge than one started the week of the deadline. Revisit if the project
  owner wants the two original commits squashed or excluded for authorship
  reasons.
- ASSUMPTION: `LICENSE` copyright line reads "The PACI-Arm Contributors"
  rather than a single GitHub handle, since the existing two commits carry a
  different author identity than the submission account. Revisit if the team
  wants a specific named copyright holder.

## Blockers

- GitHub push authentication is not yet configured in this environment (no
  `gh` CLI installed). The project owner is installing/logging in `gh`
  separately; local commits proceed and will be pushed once that's ready.
