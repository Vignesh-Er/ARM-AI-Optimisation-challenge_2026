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

TIER-1 (INT4): tools/probe_int4.py confirmed tensorflow==2.21.0's converter
never emits real INT4 tensors, so Tier-1 is exported from its INT8 baseline
.tflite by re-quantizing tier1_conv2/tier1_logits' weights to 4-bit
symmetric (range [-7,7], zero point 0) here, in Python — NOT inside the
TFLite graph. The re-quantization dequantizes each INT8 weight through its
existing (channel-wise) scale back to float, then re-quantizes to int4
using a NEW scale sized to that data (max|float weight| / 7 per group),
clamping (not wrapping) and counting clamps. The multiplier/shift for the
*output* rescale must then be recomputed from int8_scale * NEW int4_scale
/ output_scale — reusing the int8 multiplier/shift here would be wrong,
since the weight scale changed.

Read directly from the pinned CMSIS-NN headers (not assumed generically,
per section 3's own instruction to read the actual parity constraints):
arm_convolve_s4 takes `cmsis_nn_per_channel_quant_params` (an array), so
tier1_conv2 is quantized PER-CHANNEL like the INT8 conv layers. But
arm_fully_connected_s4 takes `cmsis_nn_per_tensor_quant_params` (a single
multiplier/shift, not an array) — there is no per-channel INT4 fully-
connected kernel in this CMSIS-NN checkout — so tier1_logits is
necessarily quantized PER-TENSOR (one shared int4 scale across the whole
weight tensor), which is a real constraint from the actual API, not the
generic "per-channel" the brief's section 2 assumed before this was
checked (see docs/STATUS.md for where this was verified).

Usage: python tools/export_cmsisnn.py <model.tflite> <output_header.h> --prefix {tier1,tier2}
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

from tools.int4_pack import pack_s4  # noqa: E402
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
        "input_scale": input_scale,
        "output_scale": output_scale,
        "accumulation_depth": weight.shape[1] * weight.shape[2] * weight.shape[3],  # HK*WK*C_IN
        "weight_bits": 8,
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
        "input_scale": input_scale,
        "output_scale": output_scale,
        "accumulation_depth": weight.shape[1],  # C_IN
        "weight_bits": 8,
    }


def requantize_to_int4_per_channel(int8_weight, int8_weight_scales, channel_axis=0):
    """Dequantize each channel of an INT8 baseline weight tensor back to
    float through its own INT8 scale, then re-quantize that channel to
    4-bit symmetric (range [-7,7], zero point 0) with a NEW scale sized to
    that channel's data (max|float weight| / 7). Clamps rather than wraps,
    and counts clamps per channel (section 3, step 3).
    """
    n_channels = int8_weight.shape[channel_axis]
    int4_weight = np.zeros_like(int8_weight, dtype=np.int8)
    int4_scales = []
    n_clamps = []
    for c in range(n_channels):
        idx = tuple(c if ax == channel_axis else slice(None) for ax in range(int8_weight.ndim))
        float_w = int8_weight[idx].astype(np.float64) * int8_weight_scales[c]
        max_abs = float(np.max(np.abs(float_w)))
        scale4 = max_abs / 7.0 if max_abs > 0 else 1.0
        q = np.round(float_w / scale4)
        clamped = np.clip(q, -7, 7)
        n_clamps.append(int(np.sum(q != clamped)))
        int4_weight[idx] = clamped.astype(np.int8)
        int4_scales.append(scale4)
    return int4_weight, int4_scales, n_clamps


def requantize_to_int4_per_tensor(int8_weight, int8_weight_scales_per_out_channel):
    """Same idea as requantize_to_int4_per_channel, but produces ONE scale
    for the whole tensor — required for tier1_logits, since
    arm_fully_connected_s4 takes cmsis_nn_per_tensor_quant_params (a single
    multiplier/shift), not an array; there is no per-channel INT4 fully-
    connected kernel in this CMSIS-NN checkout (verified by reading
    Include/arm_nnfunctions.h directly, not assumed — see this module's
    docstring). int8_weight_scales_per_out_channel is still per-output-
    channel (the INT8 baseline's own quantization), used only to dequantize
    back to float correctly before finding the single int4 scale.
    """
    float_weight = np.zeros(int8_weight.shape, dtype=np.float64)
    for c in range(int8_weight.shape[0]):
        float_weight[c] = int8_weight[c].astype(np.float64) * int8_weight_scales_per_out_channel[c]

    max_abs = float(np.max(np.abs(float_weight)))
    scale4 = max_abs / 7.0 if max_abs > 0 else 1.0
    q = np.round(float_weight / scale4)
    clamped = np.clip(q, -7, 7)
    n_clamps = int(np.sum(q != clamped))
    return clamped.astype(np.int8), scale4, n_clamps


def _rescale_bias(bias_int32, old_weight_scales_per_channel, new_weight_scales_per_channel):
    """TFLite's convention is bias_scale[c] = input_scale * weight_scale[c]
    (input_scale is a shared per-tensor factor, so it cancels out of the
    ratio below and never needs to appear here). Re-quantizing weights to a
    NEW per-channel scale (int4, ~18x larger than the int8 baseline in this
    architecture) without also re-quantizing the bias leaves the bias in
    the OLD scale's units while the weight*input term is now in the NEW
    scale's units — CMSIS-NN sums them as raw int32 before the single
    requantize step, so a stale bias silently corrupts every accumulator by
    a wrong, roughly-constant-per-channel amount. (Caught empirically: an
    early version of this exporter reused the INT8 baseline's bias
    unchanged, and paci_infer_t1_s4's early layers were far less saturated
    than the real INT8 baseline's own equivalent layer, i.e. exactly what
    an oversized bias relative to the shrunk weight term looks like.)
    bias_float[c] = bias_int32[c] * old_scale[c]; new_bias[c] =
    round(bias_float[c] / new_scale[c]) = round(bias_int32[c] *
    old_scale[c] / new_scale[c]).
    """
    old_weight_scales_per_channel = np.asarray(old_weight_scales_per_channel, dtype=np.float64)
    new_weight_scales_per_channel = np.asarray(new_weight_scales_per_channel, dtype=np.float64)
    ratio = old_weight_scales_per_channel / new_weight_scales_per_channel
    return np.round(bias_int32.astype(np.float64) * ratio).astype(np.int32)


def _pack_or_raise(int4_weight, layer_name):
    flat = int4_weight.reshape(-1)
    if flat.size % 2 != 0:
        raise ValueError(
            f"{layer_name}: {flat.size} int4 weight elements is odd — CMSIS-NN's "
            "packed s4 format requires an even element count (G10). Adjust this "
            "layer's channel/filter count to the nearest even value at the "
            "architecture level, not here."
        )
    return pack_s4(flat)


def export_conv_layer_s4(graph, op_idx, name):
    op = graph.op(op_idx)
    input_idx, weight_idx, bias_idx = op.Inputs(0), op.Inputs(1), op.Inputs(2)
    output_idx = op.Outputs(0)

    weight = graph.tensor_data(weight_idx)   # INT8 baseline, [C_OUT, H, W, C_IN]
    bias = graph.tensor_data(bias_idx)
    input_q = graph.tensor_quant(input_idx)
    weight_q = graph.tensor_quant(weight_idx)
    output_q = graph.tensor_quant(output_idx)

    int4_weight, int4_scales, n_clamps = requantize_to_int4_per_channel(
        weight, weight_q["scales"], channel_axis=0
    )
    packed_weight = _pack_or_raise(int4_weight, name)
    rescaled_bias = _rescale_bias(bias, weight_q["scales"], int4_scales)

    input_scale = input_q["scales"][0]
    output_scale = output_q["scales"][0]
    effective_scales = [input_scale * s4 for s4 in int4_scales]
    multipliers, shifts = per_channel_multiplier_shift(effective_scales, output_scale)

    return {
        "name": name,
        "op": "CONV_2D_S4",
        "weight_shape": list(weight.shape),
        "weight": int4_weight,  # unpacked, int4-range values stored as int8 — for Oracle B's
                                # Python reference; arithmetic is identical to any other int8
                                # weight, packing is purely a memory-format concern (G10)
        "packed_weight": packed_weight,
        "bias": rescaled_bias,
        "input_offset": -input_q["zero_points"][0],
        "output_offset": output_q["zero_points"][0],
        "multiplier": multipliers,
        "shift": shifts,
        "n_channels": len(multipliers),
        "n_clamps_per_channel": n_clamps,
        "input_scale": input_scale,
        "output_scale": output_scale,
        "accumulation_depth": weight.shape[1] * weight.shape[2] * weight.shape[3],  # HK*WK*C_IN
        "weight_bits": 4,
    }


def export_dense_layer_s4(graph, op_idx, name):
    op = graph.op(op_idx)
    input_idx, weight_idx, bias_idx = op.Inputs(0), op.Inputs(1), op.Inputs(2)
    output_idx = op.Outputs(0)

    weight = graph.tensor_data(weight_idx)   # INT8 baseline, [C_OUT, C_IN]
    bias = graph.tensor_data(bias_idx)
    input_q = graph.tensor_quant(input_idx)
    weight_q = graph.tensor_quant(weight_idx)
    output_q = graph.tensor_quant(output_idx)

    int4_weight, scale4, n_clamps = requantize_to_int4_per_tensor(weight, weight_q["scales"])
    packed_weight = _pack_or_raise(int4_weight, name)
    n_out = weight.shape[0]
    rescaled_bias = _rescale_bias(bias, weight_q["scales"], [scale4] * n_out)

    input_scale = input_q["scales"][0]
    output_scale = output_q["scales"][0]
    effective_scale = input_scale * scale4 / output_scale
    multiplier, shift = quantize_multiplier(effective_scale)

    return {
        "name": name,
        "op": "FULLY_CONNECTED_S4",
        "weight_shape": list(weight.shape),
        "weight": int4_weight,  # unpacked, see export_conv_layer_s4's comment
        "packed_weight": packed_weight,
        "bias": rescaled_bias,
        "input_offset": -input_q["zero_points"][0],
        "output_offset": output_q["zero_points"][0],
        "multiplier": multiplier,   # scalar: per-TENSOR, not per-channel
        "shift": shift,
        "n_clamps": n_clamps,
        "input_scale": input_scale,
        "output_scale": output_scale,
        "accumulation_depth": weight.shape[1],  # C_IN
        "weight_bits": 4,
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
        "input_scale": input_scale,
        "output_scale": output_scale,
        "accumulation_depth": window_size,
        "weight_bits": None,  # no weights — a pure rescale, not a learned layer
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


def render_conv_s4(layer):
    packed = layer["packed_weight"].astype(np.int8)
    total_clamps = sum(layer["n_clamps_per_channel"])
    lines = [
        f"// {layer['name']} (CONV_2D, INT4 packed two-per-byte), weight shape "
        f"{layer['weight_shape']}, per-channel clamps={layer['n_clamps_per_channel']} (total={total_clamps})",
        _c_int8_array(f"{layer['name']}_weight", packed),
        _c_int32_array(f"{layer['name']}_bias", layer["bias"]),
        _c_int32_array(f"{layer['name']}_multiplier", layer["multiplier"]),
        _c_int32_array(f"{layer['name']}_shift", layer["shift"]),
        f"#define {layer['name'].upper()}_INPUT_OFFSET {layer['input_offset']}",
        f"#define {layer['name'].upper()}_OUTPUT_OFFSET {layer['output_offset']}",
        f"#define {layer['name'].upper()}_N_CHANNELS {layer['n_channels']}",
    ]
    return "\n".join(lines)


def render_dense_s4(layer):
    packed = layer["packed_weight"].astype(np.int8)
    lines = [
        f"// {layer['name']} (FULLY_CONNECTED, INT4 packed two-per-byte, PER-TENSOR "
        f"quant -- arm_fully_connected_s4 has no per-channel variant), weight shape "
        f"{layer['weight_shape']}, clamps={layer['n_clamps']}",
        _c_int8_array(f"{layer['name']}_weight", packed),
        _c_int32_array(f"{layer['name']}_bias", layer["bias"]),
        f"#define {layer['name'].upper()}_MULTIPLIER {layer['multiplier']}",
        f"#define {layer['name'].upper()}_SHIFT {layer['shift']}",
        f"#define {layer['name'].upper()}_INPUT_OFFSET {layer['input_offset']}",
        f"#define {layer['name'].upper()}_OUTPUT_OFFSET {layer['output_offset']}",
    ]
    return "\n".join(lines)


def export_tier1(tflite_path, window_size):
    graph = TFLiteGraph(tflite_path)
    ops = graph.op_names()
    assert ops == ["EXPAND_DIMS", "CONV_2D", "RESHAPE", "EXPAND_DIMS", "CONV_2D", "RESHAPE",
                    "MEAN", "FULLY_CONNECTED"], (
        f"tier1 op sequence drifted from what GATE 2.1 verified: {ops}"
    )
    conv_idxs = find_ops(graph, "CONV_2D")
    fc_idxs = find_ops(graph, "FULLY_CONNECTED")
    mean_idx = find_ops(graph, "MEAN")[0]

    layers = [
        export_conv_layer(graph, conv_idxs[0], "tier1_conv1"),        # INT8
        export_conv_layer_s4(graph, conv_idxs[1], "tier1_conv2"),     # INT4, per-channel
        export_mean_layer(graph, mean_idx, "tier1_gap", window_size),
        export_dense_layer_s4(graph, fc_idxs[0], "tier1_logits"),     # INT4, per-tensor
    ]
    return layers


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
    renderers = {
        "MEAN": render_mean,
        "CONV_2D_S4": render_conv_s4,
        "FULLY_CONNECTED_S4": render_dense_s4,
    }
    for layer in layers:
        renderer = renderers.get(layer["op"], render_conv_or_dense)
        lines.append(renderer(layer))
        lines.append("")
    lines.append(f"#endif // {macro_guard}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="path to .tflite model")
    parser.add_argument("output", help="path to write the generated C header")
    parser.add_argument("--prefix", required=True, choices=["tier1", "tier2"], help="which model this is")
    parser.add_argument("--window", type=int, default=None, help="window size (default: config.WINDOW_SIZE)")
    args = parser.parse_args()

    if args.window is None:
        import config
        window_size = config.WINDOW_SIZE
    else:
        window_size = args.window

    if args.prefix == "tier2":
        layers = export_tier2(args.model, window_size)
    elif args.prefix == "tier1":
        layers = export_tier1(args.model, window_size)
    else:
        raise NotImplementedError(args.prefix)

    for layer in layers:
        if "n_clamps_per_channel" in layer:
            total = sum(layer["n_clamps_per_channel"])
            if total:
                print(f"  {layer['name']}: {total} int4 clamp(s) across channels {layer['n_clamps_per_channel']}")
        elif "n_clamps" in layer and layer["n_clamps"]:
            print(f"  {layer['name']}: {layer['n_clamps']} int4 clamp(s)")

    header = render_header(layers, f"PACI_{args.prefix.upper()}_WEIGHTS_H", _export_sha())
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="\n", encoding="ascii") as f:
        f.write(header)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
