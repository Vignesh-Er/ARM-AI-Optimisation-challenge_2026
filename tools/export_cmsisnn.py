# SPDX-License-Identifier: Apache-2.0
"""Task 2.3: export a trained, INT8-quantized .tflite model into C arrays
CMSIS-NN kernels can consume directly.

Layer identification walks the single subgraph's op list in order (via the
flatbuffer schema, same approach as tools/verify_bn_fold.py) rather than
matching tensor names, because TFLite's converter mangles Keras layer names
into long compound strings (confirmed empirically — see
outputs/models/shapes.json vs the actual tensor names in a converted
model) and op *input ordering* is part of the stable TFLite operator spec
(CONV_2D: [input, filter, bias]; FULLY_CONNECTED: [input, weights, bias]),
not an implementation detail.

Per-channel/per-tensor multiplier and shift are derived from each tensor's
stored float `scale` via tools/quantize_multiplier.py's QuantizeMultiplier
port — this is not "recomputing an int8 multiplier from a scale float" in
the sense G7 warns against (there is no other representation stored in the
flatbuffer to read instead); it's the canonical algorithm both the TFLite
runtime and CMSIS-NN's own TFLite integration use. Weight/bias arrays are
copied through unchanged (never renormalized), including their native
memory layout, which two empirical findings (docs/STATUS.md GATE 2.1)
confirmed already matches CMSIS-NN's expected layout for both CONV_2D
([C_OUT,H,W,C_IN]) and FULLY_CONNECTED ([C_OUT,C_IN] flattened) — verified
end to end by Oracle A (tests/test_export_cmsisnn.py), not assumed.

GlobalAveragePooling1D lowers to a MEAN op whose input/output tensors have
independently different scales (GATE 2.1 finding) — this script exports
that op's own multiplier/shift/offsets too, alongside the conv/dense layers,
for paci_infer.c's hand-written accumulate-then-requantize pooling step.

Usage: python tools/export_cmsisnn.py <model.tflite> <output_header.h> --prefix tier2
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import tensorflow as tf

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from tensorflow.lite.python import schema_py_generated as schema_fb  # noqa: E402

from tools.quantize_multiplier import quantize_multiplier  # noqa: E402
from tools.verify_bn_fold import _BUILTIN_NAME_BY_CODE, _op_name  # noqa: E402

_SHAPES_PATH = os.path.join(_PROJECT_ROOT, "outputs", "models", "shapes.json")


class TFLiteGraph:
    """Thin wrapper bundling the flatbuffer op-sequence view (for structural
    walking) with the tf.lite.Interpreter view (for tensor data/quant
    params) of the same model."""

    def __init__(self, tflite_path):
        with open(tflite_path, "rb") as f:
            self._buf = f.read()
        self._model = schema_fb.Model.GetRootAsModel(self._buf, 0)
        self._subgraph = self._model.Subgraphs(0)

        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self._tensor_details = {t["index"]: t for t in self.interpreter.get_tensor_details()}

    def op_names(self):
        return [_op_name(self._model, self._subgraph.Operators(i)) for i in range(self._subgraph.OperatorsLength())]

    def op(self, i):
        return self._subgraph.Operators(i)

    def tensor_data(self, tensor_idx):
        return self.interpreter.get_tensor(tensor_idx)

    def tensor_quant(self, tensor_idx):
        q = self._tensor_details[tensor_idx]["quantization_parameters"]
        return {
            "scales": [float(s) for s in q["scales"]],
            "zero_points": [int(z) for z in q["zero_points"]],
            "quantized_dimension": int(q["quantized_dimension"]),
        }

    def tensor_shape(self, tensor_idx):
        return [int(d) for d in self._tensor_details[tensor_idx]["shape"]]


def find_ops(graph, builtin_name):
    """Indices (in subgraph op order) of every op matching builtin_name."""
    return [i for i, name in enumerate(graph.op_names()) if name == builtin_name]


def per_channel_multiplier_shift(scales, output_scale):
    """One (multiplier, shift) pair per output channel, from
    effective_scale = input_scale * weight_scale / output_scale — the
    caller passes already-combined per-channel scales (input_scale *
    weight_scale[c]) so this just divides by output_scale and quantizes."""
    mults, shifts = [], []
    for s in scales:
        m, sh = quantize_multiplier(s / output_scale)
        mults.append(m)
        shifts.append(sh)
    return mults, shifts


def export_conv_layer(graph, op_idx, name):
    op = graph.op(op_idx)
    input_idx, weight_idx, bias_idx = op.Inputs(0), op.Inputs(1), op.Inputs(2)
    output_idx = op.Outputs(0)

    weight = graph.tensor_data(weight_idx)   # [C_OUT, H, W, C_IN], int8
    bias = graph.tensor_data(bias_idx)       # [C_OUT], int32
    input_q = graph.tensor_quant(input_idx)
    weight_q = graph.tensor_quant(weight_idx)
    output_q = graph.tensor_quant(output_idx)

    input_scale = input_q["scales"][0]
    output_scale = output_q["scales"][0]
    effective_scales = [input_scale * ws for ws in weight_q["scales"]]
    multipliers, shifts = per_channel_multiplier_shift(effective_scales, output_scale)

    return {
        "name": name,
        "op": "CONV_2D",
        "weight_shape": list(weight.shape),
        "weight": weight.astype(np.int8),
        "bias": bias.astype(np.int32),
        "input_offset": -input_q["zero_points"][0],
        "output_offset": output_q["zero_points"][0],
        "multiplier": multipliers,
        "shift": shifts,
        "n_channels": len(multipliers),
    }


def export_dense_layer(graph, op_idx, name):
    op = graph.op(op_idx)
    input_idx, weight_idx, bias_idx = op.Inputs(0), op.Inputs(1), op.Inputs(2)
    output_idx = op.Outputs(0)

    weight = graph.tensor_data(weight_idx)   # [C_OUT, C_IN], int8
    bias = graph.tensor_data(bias_idx)       # [C_OUT], int32
    input_q = graph.tensor_quant(input_idx)
    weight_q = graph.tensor_quant(weight_idx)
    output_q = graph.tensor_quant(output_idx)

    input_scale = input_q["scales"][0]
    output_scale = output_q["scales"][0]
    is_per_channel = len(weight_q["scales"]) > 1
    effective_scales = [input_scale * ws for ws in weight_q["scales"]]
    multipliers, shifts = per_channel_multiplier_shift(effective_scales, output_scale)

    return {
        "name": name,
        "op": "FULLY_CONNECTED",
        "weight_shape": list(weight.shape),
        "weight": weight.astype(np.int8),
        "bias": bias.astype(np.int32),
        "input_offset": -input_q["zero_points"][0],
        "output_offset": output_q["zero_points"][0],
        "multiplier": multipliers,
        "shift": shifts,
        "n_channels": len(multipliers),
        "is_per_channel": is_per_channel,
    }


def export_mean_layer(graph, op_idx, name, window_size):
    op = graph.op(op_idx)
    input_idx = op.Inputs(0)
    output_idx = op.Outputs(0)

    input_q = graph.tensor_quant(input_idx)
    output_q = graph.tensor_quant(output_idx)

    input_scale = input_q["scales"][0]
    output_scale = output_q["scales"][0]
    # TFLite's quantized Mean: acc = sum(x_i - input_zp); result =
    # requantize(acc, effective_scale) + output_zp, effective_scale =
    # input_scale / (window_size * output_scale) — see docs/STATUS.md
    # GATE 2.1 for how this was determined from the real op's behaviour.
    effective_scale = input_scale / (window_size * output_scale)
    multiplier, shift = quantize_multiplier(effective_scale)

    return {
        "name": name,
        "op": "MEAN",
        "input_offset": -input_q["zero_points"][0],
        "output_offset": output_q["zero_points"][0],
        "multiplier": multiplier,
        "shift": shift,
    }


def _c_int8_array(name, arr):
    flat = arr.reshape(-1)
    values = ", ".join(str(int(v)) for v in flat)
    return f"static const int8_t {name}[{flat.size}] = {{{values}}};"


def _c_int32_array(name, arr):
    flat = np.asarray(arr).reshape(-1)
    values = ", ".join(str(int(v)) for v in flat)
    return f"static const int32_t {name}[{flat.size}] = {{{values}}};"


def render_conv_or_dense(layer):
    lines = [
        f"// {layer['name']} ({layer['op']}), weight shape {layer['weight_shape']}",
        _c_int8_array(f"{layer['name']}_weight", layer["weight"]),
        _c_int32_array(f"{layer['name']}_bias", layer["bias"]),
        _c_int32_array(f"{layer['name']}_multiplier", layer["multiplier"]),
        _c_int32_array(f"{layer['name']}_shift", layer["shift"]),
        f"#define {layer['name'].upper()}_INPUT_OFFSET {layer['input_offset']}",
        f"#define {layer['name'].upper()}_OUTPUT_OFFSET {layer['output_offset']}",
        f"#define {layer['name'].upper()}_N_CHANNELS {layer['n_channels']}",
    ]
    return "\n".join(lines)


def render_mean(layer):
    return "\n".join([
        f"// {layer['name']} (MEAN / GlobalAveragePooling1D)",
        f"#define {layer['name'].upper()}_INPUT_OFFSET {layer['input_offset']}",
        f"#define {layer['name'].upper()}_OUTPUT_OFFSET {layer['output_offset']}",
        f"#define {layer['name'].upper()}_MULTIPLIER {layer['multiplier']}",
        f"#define {layer['name'].upper()}_SHIFT {layer['shift']}",
    ])


def export_tier2(tflite_path, window_size):
    graph = TFLiteGraph(tflite_path)
    ops = graph.op_names()
    assert ops == ["EXPAND_DIMS", "CONV_2D", "RESHAPE", "EXPAND_DIMS", "CONV_2D", "RESHAPE",
                    "MEAN", "FULLY_CONNECTED", "FULLY_CONNECTED", "SOFTMAX"], (
        f"tier2 op sequence drifted from what GATE 2.1 verified: {ops}"
    )
    conv_idxs = find_ops(graph, "CONV_2D")
    fc_idxs = find_ops(graph, "FULLY_CONNECTED")
    mean_idx = find_ops(graph, "MEAN")[0]

    layers = [
        export_conv_layer(graph, conv_idxs[0], "tier2_conv1"),
        export_conv_layer(graph, conv_idxs[1], "tier2_conv2"),
        export_mean_layer(graph, mean_idx, "tier2_gap", window_size),
        export_dense_layer(graph, fc_idxs[0], "tier2_dense1"),
        export_dense_layer(graph, fc_idxs[1], "tier2_logits"),
    ]
    return layers


def _export_sha(shapes_path=_SHAPES_PATH):
    with open(shapes_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def render_header(layers, macro_guard, export_sha):
    lines = [
        "// SPDX-License-Identifier: Apache-2.0",
        "// Generated by tools/export_cmsisnn.py -- DO NOT EDIT BY HAND.",
        f"#ifndef {macro_guard}",
        f"#define {macro_guard}",
        "",
        "#include <stdint.h>",
        "",
        f'#define PACI_EXPORT_SHA "{export_sha}"',
        "",
    ]
    for layer in layers:
        if layer["op"] == "MEAN":
            lines.append(render_mean(layer))
        else:
            lines.append(render_conv_or_dense(layer))
        lines.append("")
    lines.append(f"#endif // {macro_guard}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to .tflite model")
    parser.add_argument("output", help="path to write the generated C header")
    parser.add_argument("--prefix", required=True, choices=["tier2"], help="which model this is")
    parser.add_argument("--window", type=int, default=None, help="window size (default: config.WINDOW_SIZE)")
    args = parser.parse_args()

    if args.window is None:
        import config
        window_size = config.WINDOW_SIZE
    else:
        window_size = args.window

    if args.prefix == "tier2":
        layers = export_tier2(args.model, window_size)
    else:
        raise NotImplementedError(args.prefix)

    header = render_header(layers, f"PACI_{args.prefix.upper()}_WEIGHTS_H", _export_sha())
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="\n", encoding="ascii") as f:
        f.write(header)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
