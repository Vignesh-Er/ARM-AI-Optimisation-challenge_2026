# SPDX-License-Identifier: Apache-2.0
"""G2.3/G2.4 Oracle B: Tier-1 (INT4) has no TFLite interpreter to check
against — the converter never emits real INT4 tensors (Task 2.2) — so
tools/ref_cmsisnn.py's NumPy arithmetic IS the oracle, itself verified bit-
exact against the real compiled arm_nn_requantize
(tests/test_requantize_matches_cmsisnn.py) before being trusted here.
paci_infer_t1_s4() — the actual compiled function calling arm_convolve_s4/
arm_fully_connected_s4 — must match that reference exactly on held-out
windows.

Caught a real bug on the way (docs/STATUS.md GATE 2.3/2.4 Tier-1 section):
the exporter re-quantized weights to a new (~18x larger) INT4 scale but
left the INT32 bias unchanged in the OLD INT8-baseline scale's units.
CMSIS-NN sums weight-term and bias as raw int32 before a single requantize
step, so the stale bias silently corrupted every accumulator. Found by
comparing against the real INT8-baseline TFLite interpreter output (which
must stay recognisably similar to the coarser INT4 version, not
unboundedly different) rather than only checking self-consistency.
"""
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
from tools.export_cmsisnn import export_tier1  # noqa: E402

from paci_ctypes import PACI_OK, infer_t1_s4, load_lib  # noqa: E402

_TFLITE_PATH = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier1_fixture.tflite")
_N_WINDOWS = 200


def _require_fixture_model():
    if not os.path.isfile(_TFLITE_PATH):
        pytest.skip(f"{_TFLITE_PATH} not found. Run `python tools/train_fixture_models.py` first.")


@pytest.fixture(scope="module")
def lib():
    return load_lib()


def _run_reference_pipeline(window_int8, layers):
    conv1, conv2, gap, logits = layers

    x = window_int8.reshape(config.WINDOW_SIZE, 1)
    w1 = conv1["weight"].reshape(conv1["weight_shape"][0], conv1["weight_shape"][2], conv1["weight_shape"][3])
    h1 = ref.conv1d_int_reference(
        x, w1, conv1["bias"], conv1["input_offset"], conv1["output_offset"],
        conv1["multiplier"], conv1["shift"], -128, 127,
        kernel_size=w1.shape[1], padding=(w1.shape[1] - 1) // 2,
    )
    w2 = conv2["weight"].reshape(conv2["weight_shape"][0], conv2["weight_shape"][2], conv2["weight_shape"][3])
    h2 = ref.conv1d_int_reference(
        h1, w2, conv2["bias"], conv2["input_offset"], conv2["output_offset"],
        conv2["multiplier"], conv2["shift"], -128, 127,
        kernel_size=w2.shape[1], padding=(w2.shape[1] - 1) // 2,
    )
    pooled = ref.global_average_pool_s8(
        h2, gap["multiplier"], gap["shift"], gap["input_offset"], gap["output_offset"], -128, 127
    )
    n_out = logits["weight"].shape[0]
    return ref.fully_connected_int_reference(
        pooled, logits["weight"], logits["bias"], logits["input_offset"], logits["output_offset"],
        [logits["multiplier"]] * n_out, [logits["shift"]] * n_out, -128, 127,
    )


def _top2_margin(logits):
    order = np.argsort(logits)[::-1]
    return int(order[0]), int(logits[order[0]]) - int(logits[order[1]])


def test_paci_infer_t1_s4_matches_numpy_reference_exactly(lib):
    _require_fixture_model()

    physics = PhysicsModel()
    _, _, X_test, _, _, _ = generate_cnn_dataset(
        physics, n_scenarios=20, n_steps_per=500, seed=config.SEED, k=1.0
    )
    assert len(X_test) >= _N_WINDOWS

    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=_TFLITE_PATH)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    in_scale, in_zp = inp_detail["quantization"]

    layers = export_tier1(_TFLITE_PATH, config.WINDOW_SIZE)

    mismatches = []
    for idx in range(_N_WINDOWS):
        window_float = X_test[idx]
        window_int8 = np.round(window_float[:, 0] / in_scale + in_zp).astype(np.int8)

        expected_logits = _run_reference_pipeline(window_int8, layers)
        expected_class, expected_margin = _top2_margin(expected_logits)

        status, class_id, margin = infer_t1_s4(lib, list(window_int8))

        if status != PACI_OK or class_id != expected_class or margin != expected_margin:
            mismatches.append((idx, status, class_id, margin, expected_class, expected_margin))

    assert not mismatches, (
        f"{len(mismatches)}/{_N_WINDOWS} mismatches (idx, status, got_class, got_margin, "
        f"expected_class, expected_margin); first few: {mismatches[:5]}"
    )
