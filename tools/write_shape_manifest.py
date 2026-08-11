# SPDX-License-Identifier: Apache-2.0
"""Task 2.1: write outputs/models/shapes.json from the frozen Keras
architectures (phase4_tinyml/model.py). The C code (paci_core/src/paci_infer.c,
Phase 2 Task 2.4) binds its cmsis_nn_dims literals to this file, not to a
hand-copied guess of the architecture, so the two can't silently drift.

Per-layer dtype assignment is a design decision (which tensors are INT4 vs
INT8, section 2 of the brief), not something inferable from the float Keras
graph, so it's hardcoded here against the frozen architecture rather than
introspected.

Usage: python tools/write_shape_manifest.py
"""
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from phase4_tinyml.model import create_tier1_model, create_tier2_model  # noqa: E402

_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "models", "shapes.json")

# name -> target dtype for export (Task 2.2/2.3 decide int4-native vs
# int4-via-requantization; this only records INTENT, i.e. which tensors are
# meant to end up 4-bit).
_TIER1_DTYPES = {
    "tier1_conv1_int8": "int8",
    "tier1_conv2_int4": "int4",
    "tier1_logits_int4": "int4",
}
_TIER2_DTYPES = {
    "conv1": "int8",
    "conv2": "int8",
    "dense1": "int8",
    "tier2_logits": "int8",
}


def _conv_layer_entry(layer, dtype):
    c_in = layer.input.shape[-1]
    c_out = layer.output.shape[-1]
    k = layer.kernel_size[0]
    return {
        "name": layer.name,
        "op": "CONV_2D",
        "c_in": int(c_in),
        "c_out": int(c_out),
        "k": int(k),
        "pad": layer.padding,
        "output_dims_nhwc": [1, 1, int(layer.output.shape[-2]), int(c_out)],
        "dtype": dtype,
    }


def _dense_layer_entry(layer, dtype, name_override=None):
    c_in = layer.input.shape[-1]
    c_out = layer.output.shape[-1]
    return {
        "name": name_override or layer.name,
        "op": "FULLY_CONNECTED",
        "c_in": int(c_in),
        "c_out": int(c_out),
        "output_dims": [1, int(c_out)],
        "dtype": dtype,
    }


def _gap_entry(input_layer_output_shape, dtype):
    window = int(input_layer_output_shape[-2])
    c = int(input_layer_output_shape[-1])
    return {
        "name": "global_average_pool",
        "op": "AVERAGE_POOL_2D",
        "c_in": c,
        "c_out": c,
        "filter_w": window,
        "output_dims_nhwc": [1, 1, 1, c],
        "dtype": dtype,
    }


def build_tier1_manifest(window_size):
    model = create_tier1_model(input_shape=(window_size, 1))
    conv_layers = [l for l in model.layers if l.__class__.__name__ == "Conv1D"]
    dense_layers = [l for l in model.layers if l.__class__.__name__ == "Dense"]
    assert len(conv_layers) == 2 and len(dense_layers) == 1, (
        f"tier1 architecture drifted from the frozen spec: found {len(conv_layers)} "
        f"conv layers and {len(dense_layers)} dense layers, expected 2 and 1"
    )

    layers = [
        _conv_layer_entry(conv_layers[0], _TIER1_DTYPES["tier1_conv1_int8"]),
        _conv_layer_entry(conv_layers[1], _TIER1_DTYPES["tier1_conv2_int4"]),
        _gap_entry(conv_layers[1].output.shape, "int8"),
        _dense_layer_entry(dense_layers[0], _TIER1_DTYPES["tier1_logits_int4"], name_override="tier1_logits_int4"),
    ]
    return {"model": "tier1", "window_size": window_size, "n_outputs": 2, "layers": layers}


def build_tier2_manifest(window_size):
    model = create_tier2_model(input_shape=(window_size, 1), n_classes=config.N_CLASSES)
    conv_layers = [l for l in model.layers if l.__class__.__name__ == "Conv1D"]
    dense_layers = [l for l in model.layers if l.__class__.__name__ == "Dense"]
    assert len(conv_layers) == 2 and len(dense_layers) == 2, (
        f"tier2 architecture drifted from the frozen spec: found {len(conv_layers)} "
        f"conv layers and {len(dense_layers)} dense layers, expected 2 and 2"
    )

    layers = [
        _conv_layer_entry(conv_layers[0], _TIER2_DTYPES["conv1"]),
        _conv_layer_entry(conv_layers[1], _TIER2_DTYPES["conv2"]),
        _gap_entry(conv_layers[1].output.shape, "int8"),
        _dense_layer_entry(dense_layers[0], _TIER2_DTYPES["dense1"], name_override="dense1"),
        _dense_layer_entry(dense_layers[1], _TIER2_DTYPES["tier2_logits"], name_override="tier2_logits"),
    ]
    return {"model": "tier2", "window_size": window_size, "n_outputs": config.N_CLASSES, "layers": layers}


def main():
    manifest = {
        "window_size": config.WINDOW_SIZE,
        "tier1": build_tier1_manifest(config.WINDOW_SIZE),
        "tier2": build_tier2_manifest(config.WINDOW_SIZE),
    }
    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    with open(_OUTPUT_PATH, "w", newline="\n", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
