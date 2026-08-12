# SPDX-License-Identifier: Apache-2.0
"""G2.3/G2.4 Oracle A, at the actual C level: paci_infer_t2_s8() — the real
compiled function calling real CMSIS-NN kernels against the exported
weights header — must match the TFLite interpreter exactly on held-out
windows, not just the Python reference pipeline
(tests/test_export_cmsisnn.py already proved that one matches; this proves
the C port of the same arithmetic also matches, through the actual deployed
code path: arm_convolve_wrapper_s8, arm_fully_connected_per_channel_s8, and
paci_infer.c's hand-written avgpool rescale).
"""
import ctypes
import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from phase1_physics.physics_model import PhysicsModel  # noqa: E402
from phase4_tinyml.dataset import generate_cnn_dataset  # noqa: E402
from tools import ref_cmsisnn as ref  # noqa: E402
from tools.export_cmsisnn import export_tier2  # noqa: E402

from paci_ctypes import PACI_OK, infer_t2_s8, load_lib  # noqa: E402
from test_export_cmsisnn import _run_reference_pipeline  # noqa: E402

_N_WINDOWS = 200


def _require_model(path):
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found. Run `python tools/train_fixture_models.py` first.")


@pytest.fixture(scope="module")
def lib():
    return load_lib()


def _top2_margin(logits):
    best = int(logits[0])
    second = -999999
    best_idx = 0
    for i in range(1, len(logits)):
        val = int(logits[i])
        if val > best:
            second = best
            best = val
            best_idx = i
        elif val > second:
            second = val
    return best_idx, best - second


def test_paci_infer_t2_s8_matches_tflite_interpreter_exactly(lib, tier2_model_path):
    """NOTE (Task 3.0c): paci_infer_t2_s8 itself reads weights baked into
    paci_core at BUILD time (tier2_weights.h), not from tier2_model_path at
    test time — that's inherent to this project's static, no-dynamic-
    loading design (G6/section 9). Re-verifying against a Stage-B release
    model end to end is `export_cmsisnn.py ... --prefix tier2` (from the
    release .tflite) followed by `cmake --build build`, THEN this test with
    --tier2-model=<release .tflite> so the Python reference side recomputes
    from the same model the rebuilt library embeds. --tier2-model alone
    only re-points the ORACLE, not the thing being tested against it — the
    two must be kept in sync manually across a rebuild.
    """
    _require_model(tier2_model_path)
    import tensorflow as tf

    physics = PhysicsModel()
    _, _, X_test, _, _, _ = generate_cnn_dataset(
        physics, n_scenarios=20, n_steps_per=500, seed=config.SEED, k=1.0
    )
    assert len(X_test) >= _N_WINDOWS

    interp = tf.lite.Interpreter(model_path=tier2_model_path, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    in_scale, in_zp = inp_detail["quantization"]

    layers = export_tier2(tier2_model_path, config.WINDOW_SIZE)

    mismatches = []
    for idx in range(_N_WINDOWS):
        window_float = X_test[idx]
        window_int8 = np.round(window_float[:, 0] / in_scale + in_zp).astype(np.int8)

        expected_logits = _run_reference_pipeline(window_int8, layers)
        expected_class, expected_margin = _top2_margin(expected_logits)

        status, class_id, margin = infer_t2_s8(lib, list(window_int8))

        if status != PACI_OK or class_id != expected_class or margin != expected_margin:
            mismatches.append((idx, status, class_id, margin, expected_class, expected_margin))

    assert not mismatches, (
        f"{len(mismatches)}/{_N_WINDOWS} mismatches (idx, status, got_class, got_margin, "
        f"expected_class, expected_margin); first few: {mismatches[:5]}"
    )


def test_paci_infer_t2_s8_reports_bufsize_when_scratch_too_small():
    """G6: 'assert ctx->size >= required at runtime' must actually fire, not
    just exist as unreachable code — load the deliberately-undersized
    paci_core_tiny_scratch build (16-byte scratch, CMakeLists.txt) and
    confirm paci_infer_t2_s8 returns PACI_E_BUFSIZE instead of overrunning
    the buffer."""
    candidates = [
        os.path.join(_PROJECT_ROOT, "build", "paci_core", name)
        for name in ("libpaci_core_tiny_scratch.dll", "libpaci_core_tiny_scratch.so")
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        pytest.skip("paci_core_tiny_scratch not built; run cmake --build build --target paci_core_tiny_scratch")

    tiny_lib = ctypes.CDLL(path)
    tiny_lib.paci_infer_t2_s8.restype = ctypes.c_int
    tiny_lib.paci_infer_t2_s8.argtypes = [
        ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32),
    ]

    window = (ctypes.c_int8 * config.WINDOW_SIZE)(*([0] * config.WINDOW_SIZE))
    class_id = ctypes.c_int8()
    margin = ctypes.c_int32()
    status = tiny_lib.paci_infer_t2_s8(window, ctypes.byref(class_id), ctypes.byref(margin))

    PACI_E_BUFSIZE = -4
    assert status == PACI_E_BUFSIZE
