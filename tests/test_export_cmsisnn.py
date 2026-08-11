# SPDX-License-Identifier: Apache-2.0
"""G7/Oracle A: tools/export_cmsisnn.py's extracted weights/bias/multiplier/
shift, run through tools/ref_cmsisnn.py's int8 arithmetic primitives, must
match the real TFLite interpreter's pre-softmax logits EXACTLY (element for
element, not "close") on held-out windows. This is the Tier-2 oracle — G2.3
requires 200 windows; this test uses 200.

Deliberately reads the pre-softmax logits tensor directly
(experimental_preserve_all_tensors=True) rather than the softmax output:
softmax is monotonic so an argmax-only check could pass with logits that are
merely close, not exact, which is not what G7 asks for.
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
from tools.export_cmsisnn import TFLiteGraph, export_tier2, find_ops  # noqa: E402

_TFLITE_PATH = os.path.join(_PROJECT_ROOT, "outputs", "models", "tier2_fixture.tflite")
_N_WINDOWS = 200


def _require_fixture_model():
    if not os.path.isfile(_TFLITE_PATH):
        pytest.skip(
            f"{_TFLITE_PATH} not found. Run `python tools/train_fixture_models.py` first."
        )


def _run_reference_pipeline(window_int8, layers):
    conv1, conv2, gap, dense1, logits = layers

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
    d1 = ref.fully_connected_int_reference(
        pooled, dense1["weight"], dense1["bias"], dense1["input_offset"], dense1["output_offset"],
        dense1["multiplier"], dense1["shift"], -128, 127,
    )
    return ref.fully_connected_int_reference(
        d1, logits["weight"], logits["bias"], logits["input_offset"], logits["output_offset"],
        logits["multiplier"], logits["shift"], -128, 127,
    )


def test_tier2_reference_pipeline_matches_tflite_interpreter_exactly():
    _require_fixture_model()
    import tensorflow as tf

    physics = PhysicsModel()
    _, _, X_test, _, _, _ = generate_cnn_dataset(
        physics, n_scenarios=20, n_steps_per=500, seed=config.SEED, k=1.0
    )
    assert len(X_test) >= _N_WINDOWS, f"only {len(X_test)} test windows available, need {_N_WINDOWS}"

    interp = tf.lite.Interpreter(model_path=_TFLITE_PATH, experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    in_scale, in_zp = inp_detail["quantization"]

    # Structural lookup (same approach as export_cmsisnn.py), not name
    # matching: tensor names like "tier2_logits_1/BiasAdd" also match the
    # BIAS *weight* constant, not the runtime activation output — an
    # earlier version of this test grabbed that by mistake via substring
    # search and got a shape-[5] constant instead of the real per-window
    # logits tensor. The output of the LAST FULLY_CONNECTED op is
    # unambiguous.
    graph = TFLiteGraph(_TFLITE_PATH)
    last_fc_op_idx = find_ops(graph, "FULLY_CONNECTED")[-1]
    logits_tensor_idx = graph.op(last_fc_op_idx).Outputs(0)

    layers = export_tier2(_TFLITE_PATH, config.WINDOW_SIZE)

    mismatches = []
    for idx in range(_N_WINDOWS):
        window_float = X_test[idx]
        window_int8 = np.round(window_float[:, 0] / in_scale + in_zp).astype(np.int8)

        interp.set_tensor(inp_detail["index"], window_int8.reshape(1, config.WINDOW_SIZE, 1).astype(np.int8))
        interp.invoke()
        tflite_logits = interp.get_tensor(logits_tensor_idx)[0]

        my_logits = _run_reference_pipeline(window_int8, layers)
        if not np.array_equal(my_logits, tflite_logits):
            mismatches.append((idx, my_logits.tolist(), tflite_logits.tolist()))

    assert not mismatches, (
        f"{len(mismatches)}/{_N_WINDOWS} logit mismatches against the TFLite interpreter; "
        f"first few: {mismatches[:3]}"
    )
