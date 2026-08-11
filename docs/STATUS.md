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
- [x] Phase 1 — Unified C core
- [x] Phase 2 — CMSIS-NN integration and export
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

## Phase 1 — Unified C core (done 2026-08-11)

`paci_core/` implements the exact interface contract from section 8 of the
brief (`paci_core.h` is copied verbatim, byte-for-byte except for cosmetic
`#define` column alignment). D2-D5 are satisfied by construction:

- `src/paci_physics.c` — Deal-Grove etch-rate model, ported term-for-term
  from `phase1_physics/physics_model.py:PhysicsModel.etch_rate`.
- `src/paci_ekf.c` — scalar EKF predict+update, ported from
  `phase2_ekf/ekf.py`; implements G13 (checks `P_pred`/`S` before dividing,
  resets `P` to `PACI_P0_VAR` and increments `health_resets` on violation,
  never lets NaN through — verified by `tests/test_ekf_health.py`, which
  forces the reset path with a deliberately negative `R` rather than relying
  on the fault dataset to happen to trigger it, which it never does under
  healthy Q/R: `health_resets` is 0 for all 2000 steps of the golden trace).
- `src/paci_ring.c` — fixes D2 (`paci_ring_read` linearises chronologically,
  oldest-first, from wherever `head` physically is) and D3 (`total` is
  `uint32_t`, `head` is `uint8_t` bounded to `[0,32)` by explicit modulo, so
  neither overflows the way the old `uint8_t buffer_index` did). Verified in
  `tests/test_ring_buffer.py`; spot-checked that the *raw* physical buffer
  (what the old bug would have handed to inference) is genuinely
  non-monotonic at a nonzero phase while `paci_ring_read`'s output is —
  confirms the test isn't tautological.
- `src/paci_cascade.c` — `paci_init`/`paci_step` orchestration. Tier 0 (EKF +
  NIS/watchdog/burn-in gating) is fully implemented and tested. Tier 1/2
  dispatch — actually invoking CMSIS-NN inference and advancing
  `tier_reached` past `PACI_TIER_0_EKF` — is Phase 2 work, once
  `paci_infer_t1_s4`/`paci_infer_t2_s8` exist to invoke; `wake_reason`
  already reports the real Tier-0 evidence (NIS/watchdog/burn-in) that would
  trigger that escalation.

`tools/gen_params.py` generates `paci_core/include/paci_params.h` from
`config.py` (D4); `tests/test_params_sync.py` regenerates it to a scratch
path and diffs against the committed copy — confirmed this fails when
`config.py` drifts (temporarily set `TAU = 0.1` to match the old buggy C
value, test failed, reverted).

`tests/test_bitexact.py` (G1) replays the standard 2000-step/seed-42 dataset
through `paci_core` via ctypes and checks it against a golden trace
(`tests/golden/ekf_trace_seed42.json`, recorded once via
`tools/record_golden_trace.py`) to within 1 ULP of float32.

The old `Core/` directory (containing the D2/D3/D4/D5 bugs) has been removed
— `paci_core/` supersedes it entirely for physics/EKF/ring/Tier-0 logic. An
equivalent embedded main-loop demo will be rebuilt on the real API in a
later phase (bench harness / demo script), not restored as-is.

Build/test toolchain confirmed on this machine: gcc 13.2.0 (MinGW-w64),
cmake 4.4.2 (installed via `pip install cmake`, since no system CMake was
present — `pip`'s Scripts dir needs to be on `PATH`), Python 3.12.10 with
numpy/scipy/pytest already installed. `tensorflow` is **not** installed yet —
flagged as a Phase 2/4 blocker (needed to train/quantize the CNN and export
TFLite). 14/14 tests pass via both `pytest tests/` and `ctest --test-dir
build`.

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
- ASSUMPTION (`paci_core/include/paci_core.h`, top of file): the roadmap
  (section 7, Phase 1) lists `paci_types.h` alongside `paci_core.h` and
  `paci_params.h` as files to create, but section 8's exact-header mandate
  puts every typedef directly in `paci_core.h`. Splitting the typedefs into a
  separate `paci_types.h` would mean `paci_core.h`'s visible contents no
  longer match the mandated text exactly. Chose to keep everything in
  `paci_core.h` as specified and skip a redundant `paci_types.h`. Revisit if
  the project owner specifically wants the split (e.g. for a future non-C
  binding that only wants the types).
- ASSUMPTION (`paci_core/src/paci_cascade.c`, `PACI_PROVISIONAL_INPUT_SCALE`):
  the CNN's true int8 input scale/zero-point are decided by the TFLite
  converter when Phase 2 trains and quantizes the model; there is no trained
  model yet to extract them from (G7). Ring-buffer samples are quantized
  under a provisional fixed scale (0.25/LSB on the same normalized signal
  `phase4_tinyml/dataset.py` trains on) sized not to saturate int8 for the
  current dataset's worst case. Must be replaced by the real exported scale
  in Phase 2 — any window recorded before that point is provisional, not
  representative, and should not be used as calibration data.
- ASSUMPTION (`config.py`, `NORM_SCALE`): added this constant (was previously
  a hardcoded `50.0` literal only in `phase4_tinyml/dataset.py`) and updated
  `dataset.py` to import it, so the C ring-buffer quantizer and the Python
  training normalization can't silently diverge the way `TAU`/`Q_VAR` did
  (D4's root cause, generalised). Small, deliberate edit to carried-over
  Python, not a rewrite.

## Phase 2 — in progress

- `tensorflow==2.21.0` installed successfully in this environment; the
  earlier blocker is resolved.
- CMSIS-NN and CMSIS-DSP vendored as git submodules under `third_party/`,
  pinned to the exact tags section 9 specifies: CMSIS-NN `v7.0.0`
  (`22080c68`), CMSIS-DSP `v1.17.1` (`4b4fa8ff`) — both tags confirmed to
  exist on the upstream remotes before pinning.
- Checked the pinned v7.0.0 headers directly rather than coding from memory
  of an older CMSIS-NN API (the brief specifically warns v6/v7 have
  incompatible signatures): confirmed `arm_convolve_s4` /
  `arm_convolve_s4_get_buffer_size`, `arm_convolve_wrapper_s8` (+
  `_get_buffer_size`/`_mve`/`_dsp` variants), `arm_fully_connected_s4`,
  `arm_fully_connected_s8` (+ buffer-size variants), `arm_softmax_s8` all
  exist in `Include/arm_nnfunctions.h` with the signatures Phase 2 needs.
  There is also `arm_convolve_even_s4`, a stricter variant requiring an even
  kernel-size product — relevant to G10's packing constraint, to be decided
  once the Tier-1 model architecture is fixed.
- Verified G8's anchor directly in source (not just taking the brief's
  paraphrase on faith): `arm_nn_requantize` in
  `third_party/CMSIS-NN/Include/arm_nnsupportfunctions.h:1577-1591`, and
  under `CMSIS_NN_USE_SINGLE_ROUNDING` it is exactly
  `result = new_val >> (total_shift - 1); result = (result + 1) >> 1;`
  — round-half-up, confirming the bias-budget analysis in Phase 5 has real
  ground to stand on.

## Phase 2 (continued) — Task 2.0-2.2 progress (2026-08-11)

- **Task 2.0 (window probe)**: added a `k` severity multiplier to
  `phase1_physics/synthetic_data.py:inject_fault()`/`generate_full_dataset()`.
  k=1.0 reproduces the original fixed-magnitude gas_leak/equipment_drift/
  unexpected_deviation faults byte-for-byte against a captured pre-edit
  baseline (`tests/golden/synthetic_data_baseline.npz`,
  `tests/test_fault_injection_k1_matches_baseline.py`). `sensor_fault` was
  deliberately redefined from flatline-to-0.0 (a ~125-sigma jump, physically
  implausible for a stuck sensor) to stuck-at-last-reading, per instruction
  — this is NOT byte-for-byte equivalent at any k, including k=1.0, and the
  test documents that as the one intentional exception. Regenerated
  `tests/golden/ekf_trace_seed42.json` afterward since the dataset generator
  it was recorded against changed (not a paci_core regression — confirmed
  by re-running the full test suite, 19/19 pass).
  - First sweep run produced a **meaningless result**: window=64 and
    window=128 both came back at a trivial 1.0 "balanced accuracy" with
    100%-Normal predictions. Root cause: `tools/probe_window.py` initially
    reused `n_steps_per=500` at every window size, but
    `generate_full_dataset`'s `fault_duration = max(20, n_steps_per // 20)`
    depends only on `n_steps_per`, not window size, and `extract_windows`
    labels each window by **majority vote**. At `n_steps_per=500`,
    `fault_duration=25`, which is under 50% of a 64- or 128-sample window,
    so no window could ever get a nonzero majority label — every window in
    those cells was labeled Normal by construction, not because the fault
    was undetectable. Fixed by scaling `n_steps_per = 20 * window_size` so
    `fault_duration` tracks `window_size` across the sweep. Flagging this
    for Phase 4: the real severity ladder needs to choose fault durations
    deliberately relative to whatever `PACI_WINDOW_SIZE` ends up frozen at,
    not inherit whatever `n_steps_per` happens to be used elsewhere.
  - Sweep re-running with the fix; decision + `outputs/probe/window_sweep.json`
    to follow in the next STATUS.md update.
- **Task 2.2 (INT4 export probe)**: ran `tools/probe_int4.py` against
  `tensorflow==2.21.0`. Setting the converter's low-bit-QAT attribute is
  accepted (`used_int4_api=True`) but the converted flatbuffer's tensor
  types are only `{INT32, INT8}` — **no INT4 tensors emitted**. Matches the
  brief's expectation (tensorflow/tensorflow#64193). Taking the fallback
  branch: Tier-1 converts to standard full-integer INT8, and
  `tools/export_cmsisnn.py` (Task 2.3) re-quantizes the designated layers to
  4-bit symmetric per-channel on top of that.
  - The int4 nibble pack/unpack order was **verified from CMSIS-NN's own
    test-vector generator**, not derived by reading the DSP/MVE unpack
    intrinsics (`read_and_pad_s4` in `Include/arm_nnsupportfunctions.h`,
    which is a SIMD register-loading shuffle, not the wire format):
    `third_party/CMSIS-NN/Tests/UnitTest/conv_settings.py:299-300` (identical
    in `fully_connected_settings.py:212-213`) packs
    `(0xf0 & (v1 << 4)) | (v0 & 0xf)` — even index in the low nibble, odd
    index in the high nibble. Implemented in `tools/int4_pack.py`, round-trip
    tested in `tests/test_int4_packing.py` (full int4 range + 5 random
    seeds + the exact byte value for a known pair + rejection of odd-length/
    out-of-range input) — 10/10 pass.

## GATE 2.0 — window size frozen (2026-08-11)

Amendments applied before Task 2.1 (project-owner instruction):

**A. Labeling rule.** `phase4_tinyml/dataset.py:extract_windows()` gained a
`rule` parameter: `"last"` (new default, `y[i] = labels[i + window_size - 1]`,
the label of the most recent sample) and `"majority"` (the original scheme,
kept reachable only so the sweep could report both columns once as evidence).
`"last"` is used everywhere from this point on. **Why:** the deployed cascade
never sees future samples — `paci_step` only ever knows "now" and everything
before it — so `"last"` is what the model is actually trained to replicate.
It's also the only rule under which Phase 4's detection-latency metric is
coherent: `"majority"` only labels a window "faulty" once the fault already
covers half the window, which silently adds `window_size / 2` steps of
phantom latency having nothing to do with detection quality. **Phase 4's
detection-latency definition must use `"last"` too, or the metric measured
there is not the same thing this window-size decision was calibrated
against.**

**B. Decision rule**, replacing the earlier fixed-0.75-threshold version:
smallest `W` in `{32,64,128}` such that balanced accuracy at `k=0.1` for `2W`
exceeds `W` by less than 5 points (i.e. doubling stops paying for itself).

**C. `tools/ref_cmsisnn.py` directed test cases** (`tests/test_requantize_matches_cmsisnn.py`):
exact-tie construction (`make_tie_case`) for 40 shift values (`-20..19`), 100
positive-odd-R cases each mirrored to negative (200+/shift, 54 tests total,
all passing against the real compiled `arm_nn_requantize`), plus a
sign-asymmetry invariant check (`result(R) + result(-R) == 1`, the
round-toward-+infinity signature — a magnitude-symmetric scheme would give
0). `tools/probe_requantize_bias.py` emits `outputs/probe/requantize_bias.json`
for Phase 5 to import. First version of that script used independently
random `(val, multiplier, shift)`, which let the "exact value" reach ~1e12
(unrealistic — real CMSIS-NN multiplier/shift pairs are calibrated together,
not independent) and reported a meaningless ~1e12 "bias"; fixed by deriving
shift from val/multiplier so the exact value lands near a representative
magnitude. **Finding, not assumed**: the bulk (mostly non-tie) population
bias is ~0.001-0.002, close to zero — NOT +0.5. This does not contradict the
deterministic +0.5-at-ties result; the function is two sequential shifts
(floor, then round-half-up) whose opposite directional biases cancel for a
generic fractional remainder, leaving the full +0.5 asymmetry visible only
at exact ties. **Phase 5's bias budget must anchor on the tie-conditional
+0.5, not the bulk mean** — the bulk mean understates risk for any
computation whose remainder distribution isn't close to uniform. Full
reasoning is in the JSON's own `note` field (`docs never contain a
hand-typed number` — Phase 5 reads the file, not a paraphrase of it here).

**Window sweep results** (`outputs/probe/window_sweep.json`, `"last"` rule,
`k=0.1`, the decision severity): window=32 → 22.7% balanced accuracy,
window=64 → 28.6%, window=128 → 30.7%. Improvement 64→128 is 2.0 points,
under the 5-point margin, so **PACI_WINDOW_SIZE = 64** — set in `config.py`,
`paci_core.h`'s `#define` (with an `// ASSUMPTION:` comment at the decision
site — this supersedes the original brief's illustrative `32`), and
`paci_params.h` regenerated. All window-size-dependent tests
(`tests/paci_ctypes.py`, `tests/test_ring_buffer.py`) updated to reference
the frozen constant rather than a hardcoded `32`; full suite re-verified
(82/82 pass) after the change.

**Honest caveat, not smoothed over**: absolute detection quality at k=0.1 is
weak across every window size — per-class recall at window=64/k=0.1 is
`[0.994, 0.0, 0.0, 0.437, 0.0]` (Normal, Sensor Fault, Gas Leak, Equipment
Drift, Unexpected Deviation): the classifier detects almost nothing at 10%
of the original fault magnitude except partial Equipment Drift. The decision
rule is about *diminishing returns from a larger window*, not an absolute
performance bar, so procedurally the window=64 choice stands — but Phase 4's
severity ladder should not assume k=0.1 is an achievable "hard end" for
every fault class with this window/architecture; it may need a shallower
floor for Sensor Fault/Gas Leak/Unexpected Deviation specifically, informed
by this same sweep data rather than picked arbitrarily.

## GATE 2.1 — fixture models trained, BN fold verified (2026-08-11)

`tools/train_fixture_models.py` generates the fixture dataset (window=64,
k=1.0, `n_scenarios=20`, `n_steps_per=500`) and trains both models via
`phase4_tinyml/train.py`. `tools/verify_bn_fold.assert_bn_folded()` confirms
no unfused `MUL`/`ADD` immediately follows any `CONV_2D` for either model.
`tools/write_shape_manifest.py` wrote `outputs/models/shapes.json`.

Sizes: `tier2_fixture.tflite` 11.80 KB (matches the ~11.78 KB the old,
now-deleted README claimed for the original ungated model — same budget,
now honestly earned). `tier1_fixture.tflite` 6.90 KB as an **INT8 baseline**
— over the "<3 KB" Tier-1 target as-is, but that target is for the INT4
*deployed* footprint (Task 2.2/2.3); the INT8 baseline being larger than
the eventual INT4-packed size is expected, not a budget violation yet.
Actual post-packing size will be checked at export time; if still over
budget, the fallback is reducing channel counts to the next even value
(section 2), never changing the now-frozen window.

**Two real findings from inspecting the actual converted `.tflite` files
directly** (`tools/verify_bn_fold.op_sequence` + `tf.lite.Interpreter`),
not assumed from the architecture alone — both materially change the
Task 2.3/2.4 plan:

1. **`GlobalAveragePooling1D` lowers to a `MEAN` op, not `AVERAGE_POOL_2D`**,
   and critically **its input and output tensors have independently
   *different* scales** (tier2: input scale 0.1390902 → output scale
   0.01018135, ~13.7x ratio; zero-points happened to both be -128 here but
   that's not guaranteed either). `arm_avgpool_s8`'s signature
   (`cmsis_nn_pool_params`: stride/padding/activation only, no offset or
   multiplier/shift fields) assumes input and output share one scale —
   it is not a valid substitute for what these models actually do at that
   step. TFLite's own quantized `Mean` kernel instead accumulates the raw
   int8 sum, requantizes it with a multiplier/shift derived from
   `input_scale / (window_size * output_scale)` (via `QuantizeMultiplier`),
   then adds `output_zero_point` — exactly the pattern
   `tools/ref_cmsisnn.py:global_average_pool_s8` already implements (written
   before this was confirmed, from reasoning about what a quantized mean
   *should* do — turned out to match TFLite's actual kernel). Task 2.4's
   `paci_infer.c` will implement this directly (~10 lines, same
   accumulate-then-`arm_nn_requantize` pattern CMSIS-NN's own kernels use
   throughout) rather than calling a CMSIS-NN pooling kernel that doesn't
   fit — verified against Oracle A (the real TFLite interpreter output),
   not asserted by construction.
2. **Every Dense (`FULLY_CONNECTED`) weight tensor is per-channel
   quantized**, not per-tensor (`tier2_logits` weights: 5 scales for 5
   output channels; `dense1` weights: 32 scales for 32 output channels;
   confirmed via `quantization_parameters.scales` length matching the
   output-channel count exactly, `quantized_dimension=0`). Task 2.4 must
   use `arm_fully_connected_per_channel_s8` (`cmsis_nn_per_channel_quant_params`,
   an array of multiplier/shift), not `arm_fully_connected_s8` (which takes
   a single per-tensor `cmsis_nn_per_tensor_quant_params`) — using the
   per-tensor variant against per-channel-quantized weights would silently
   apply only the first channel's scale to every output.

## GATE 2.2 (done in GATE 2.0's commit)

INT4 probe already run and recorded — see the GATE 2.0 section above.

## GATE 2.3/2.4 — Tier-2 (2026-08-11)

`tools/export_cmsisnn.py` walks the TFLite flatbuffer's op sequence
structurally (same schema-walking approach as `tools/verify_bn_fold.py`),
not by tensor-name matching — op input ordering (`CONV_2D`: `[input,
filter, bias]`; `FULLY_CONNECTED`: `[input, weights, bias]`) is part of the
stable TFLite operator spec, while converted tensor names are compound and
implementation-detail-shaped (e.g. `tier2_logits_1/MatMul;tier2_logits_1/BiasAdd`).
Per-channel multiplier/shift come from `tools/quantize_multiplier.py`'s
`QuantizeMultiplier` port applied to each tensor's stored float scale.
Weight/bias arrays are copied through with no transpose — confirmed
correct only by the oracle passing, not assumed.

**Oracle A, done in two stages**, both passing:
1. `tests/test_export_cmsisnn.py`: exported weights run through
   `tools/ref_cmsisnn.py`'s Python arithmetic primitives match the real
   TFLite interpreter's pre-softmax logits **exactly**, element for
   element, on 200 held-out windows (reads the actual logits tensor via
   `experimental_preserve_all_tensors`, not the softmax output — softmax
   is monotonic, so an argmax-only check could pass on merely-close
   logits, which isn't what G7 asks for). Caught one bug on the way: the
   first draft searched for the logits tensor by substring name match
   (`"BiasAdd" in name and "logits" in name`), which matched the bias
   *weight constant* (also named `.../logits_1/BiasAdd`) instead of the
   runtime activation tensor — fixed by using the same structural op-walk
   the exporter uses (`Outputs(0)` of the last `FULLY_CONNECTED` op).
2. `tests/test_infer_t2.py`: the actual compiled `paci_infer_t2_s8()` —
   real `arm_convolve_wrapper_s8`, real `arm_fully_connected_per_channel_s8`,
   linked against the full vendored CMSIS-NN library — matches the TFLite
   interpreter exactly (class_id AND margin) on the same 200 windows,
   through ctypes against the built `.dll`. This is the actual gate
   requirement (paci_infer_t2_s8 itself, not a Python stand-in for it).

`paci_core/src/paci_infer.c`: `arm_convolve_wrapper_s8` for both conv
layers (`*_get_buffer_size` checked against the static scratch buffer at
runtime, `PACI_E_BUFSIZE` on overflow — G6), `arm_fully_connected_per_channel_s8`
for both dense layers (per-channel, not the per-tensor variant — GATE 2.1's
finding), and a **hand-written** accumulate-then-`arm_nn_requantize` global
average pool (not `arm_avgpool_s8`, which has no rescale parameters and
can't reproduce the real Mean op's differing input/output scales — GATE
2.1's other finding). `paci_infer_t2_s8`/`paci_infer_t1_s4` are declared in
`paci_internal.h` rather than the exact-contract `paci_core.h` (section 8
predates these two functions; extending that file would break its own
byte-for-byte mandate).

G6 test (`test_paci_infer_t2_s8_reports_bufsize_when_scratch_too_small`):
`PACI_SCRATCH_BYTES` is overridable via compile definition; a second CMake
target (`paci_core_tiny_scratch`, 16-byte scratch) confirms
`PACI_E_BUFSIZE` actually fires rather than existing as unreachable code.

`tier2_weights.h` is generated (`paci_core/include/`, gitignored like the
`.tflite`/`.h5` it derives from — regenerate via
`tools/export_cmsisnn.py outputs/models/tier2_fixture.tflite
paci_core/include/tier2_weights.h --prefix tier2`); `paci_infer.c` guards
on `__has_include("tier2_weights.h")` so the library still builds (Tier-2
inference reporting `PACI_E_QUANT`, "not built") if that step is skipped.

92/92 tests pass.

## GATE 2.3/2.4 — Tier-1 (INT4), real C inference, verified bit-exact (2026-08-11)

`tools/export_cmsisnn.py` extended for Tier-1's INT4 re-quantization
(section 3's confirmed fallback branch — Task 2.2 found the TFLite
converter never emits real INT4). For each designated layer: dequantize
the INT8-baseline weight back to float through its own (per-channel) INT8
scale, re-quantize to 4-bit symmetric (range [-7,7], zero point 0, clamped
not wrapped — zero clamps occurred in practice, expected since the scale
is deliberately sized to the channel's own max magnitude), pack via
`tools/int4_pack.py` (nibble order verified against CMSIS-NN's own test-
vector generator, GATE 2.0), recompute multiplier/shift from the NEW scale
via `tools/quantize_multiplier.py` (never reused the INT8 multipliers).

**Real API constraint found by reading the headers, not assumed**
(section 3's own instruction): `arm_convolve_s4` takes
`cmsis_nn_per_channel_quant_params` (an array — `tier1_conv2` is quantized
per-channel, as the brief's section 2 assumed), but `arm_fully_connected_s4`
takes `cmsis_nn_per_tensor_quant_params` (a single multiplier/shift — there
is no per-channel INT4 fully-connected kernel in this CMSIS-NN checkout),
so `tier1_logits` is necessarily quantized **per-tensor**, one shared int4
scale across the whole weight matrix, not per-channel. Documented at the
decision site in `tools/export_cmsisnn.py` and in `paci_infer.c`.

**Real bug found and fixed, not just theoretical risk**: the first version
reused the INT8-baseline's `int32` bias unchanged for the INT4 layers.
TFLite's convention is `bias_scale = input_scale * weight_scale`; since
re-quantizing the weight to INT4 changed `weight_scale` by roughly 18x in
this architecture, the untouched bias was now off by that same ~18x
relative to the (correctly rescaled) weight-term, and CMSIS-NN sums weight
-term and bias as raw `int32` before a single `arm_nn_requantize` call — so
every accumulator was silently corrupted. Symptom: `paci_infer_t1_s4`'s
early layers were dramatically *less* saturated than the real INT8-baseline
TFLite interpreter's equivalent layer on the same input (an oversized bias
was overwhelming a correctly-shrunk weight term) — caught by comparing
against the real INT8 baseline's own intermediate tensors
(`experimental_preserve_all_tensors`), not by the INT4 path merely being
self-consistent with itself. Fixed by `_rescale_bias()`: re-derive the bias
from its dequantized float value under the new weight scale, the same
"dequantize through the old scale, requantize under the new one" pattern
used for the weights themselves.

Oracle B (no TFLite interpreter exists for real INT4, so
`tools/ref_cmsisnn.py`'s NumPy primitives — already proven bit-exact
against the real compiled `arm_nn_requantize`, GATE 2.0 — are the ground
truth): `tests/test_infer_t1.py` — the actual compiled `paci_infer_t1_s4()`,
real `arm_convolve_s4` and `arm_fully_connected_s4` kernels linked in,
matches the NumPy reference exactly (class_id and margin) on 200 held-out
windows via ctypes.

`paci_core/src/paci_infer.c` restructured so the shared helpers
(`paci_conv1d_s8`, `paci_global_avgpool_s8`, `paci_top2_margin`) compile
under `#if PACI_HAVE_TIER1 || PACI_HAVE_TIER2` rather than nested inside
Tier-2's block only — an earlier draft had Tier-1's code calling functions
that only existed when Tier-2's weights header was also present, which
happened to work in this repo (both tiers are always exported together)
but was structurally wrong.

Deployed Tier-1 weight+bias footprint: 216 bytes (0.21 KB), well under the
<3 KB budget (section 2) — no need to shrink channel counts.

93/93 tests pass. `tools/check_no_fixture_in_results.py` (G2.5): clean, no
fixture references in `outputs/reports/`, `outputs/bench/`, or `README.md`.

**Phase 2 complete** (G2.0 through G2.5). Remaining before Phase 3: none —
Phase 3 (measurement harness) is next.

## Two-stage model plan (fixture vs release)

Per project-owner instruction: train fixture models now (throwaway weights,
permanent shapes) to unblock the exporter/packing/buffer-sizing/bit-exact
tests, which all depend on dimensions rather than accuracy. Release models
(same architecture, retrained on Phase 4's severity-laddered dataset) come
later with no C changes required, by construction. Fixture artifacts are
named `*_fixture.tflite`/`*_fixture.h`; release artifacts `*_release.*`.
`tools/check_no_fixture_in_results.py` greps `outputs/reports/`,
`outputs/bench/`, and `README.md` for "fixture" and fails the build if found
— a fixture number must never reach a results artifact.

## Blockers

- GitHub push authentication is not yet configured in this environment (no
  `gh` CLI installed). The project owner is installing/logging in `gh`
  separately; local commits proceed and will be pushed once that's ready.
- ~~`tensorflow` not installed~~ — resolved: `pip install tensorflow` succeeded,
  `tensorflow==2.21.0` importable. No remaining blocker for Phase 2 training.
