# SPDX-License-Identifier: Apache-2.0
"""Bit-accurate NumPy reference for CMSIS-NN's requantization arithmetic.

This exists because Tier-1 (INT4) has no TFLite interpreter oracle to check
against — the converter never emits real INT4 tensors (Task 2.2), and the
INT4 layers are a from-scratch re-quantization this project does itself.
Oracle B for Tier-1 (Task 2.3) is: does paci_infer_t1_s4() match THIS
module, exactly, on held-out windows.

Reused in Phase 5 (the rounding-bias budget): the round-half-up bias term
this module reproduces is the same one that analysis quantifies.

Implements arm_nn_requantize under CMSIS_NN_USE_SINGLE_ROUNDING exactly as
verified in the pinned checkout,
third_party/CMSIS-NN/Include/arm_nnsupportfunctions.h:1577-1591:

    const int64_t total_shift = 31 - shift;
    const int64_t new_val = val * (int64_t)multiplier;
    int32_t result = new_val >> (total_shift - 1);
    result = (result + 1) >> 1;
"""
import numpy as np


def _wrap_int64(x):
    x &= 0xFFFFFFFFFFFFFFFF
    return x - 0x10000000000000000 if x >= 0x8000000000000000 else x


def _wrap_int32(x):
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def requantize(val, multiplier, shift):
    """arm_nn_requantize, CMSIS_NN_USE_SINGLE_ROUNDING path. val, multiplier,
    shift may be Python ints or numpy int arrays. The C source is:

        const int64_t total_shift = 31 - shift;
        const int64_t new_val = val * (int64_t)multiplier;
        int32_t result = new_val >> (total_shift - 1);
        result = (result + 1) >> 1;
        return result;

    The middle line's assignment to `int32_t result` TRUNCATES the shifted
    64-bit value to 32 bits (two's-complement wraparound) — this is not
    optional bookkeeping, it materially changes the output whenever the
    shifted value doesn't fit in 32 bits (verified against the actual
    compiled arm_nn_requantize via tools/native/requantize_shim.c: an
    earlier version of this function that used Python's arbitrary-precision
    ints throughout, without this truncation, mismatched the real function
    on ~100% of large-shift cases in a 20000-sample cross-check). Python
    ints are arbitrary precision, so each C-typed intermediate is wrapped
    explicitly to reproduce that truncation.
    """
    val = int(val)
    multiplier = int(multiplier)
    shift = int(shift)

    total_shift = 31 - shift
    new_val = _wrap_int64(val * multiplier)
    result = _wrap_int32(new_val >> (total_shift - 1))
    result = _wrap_int32((result + 1) >> 1)
    return np.int32(result)


def requantize_array(vals, multipliers, shifts):
    """Vectorised requantize over per-channel multiplier/shift arrays.
    vals, multipliers, shifts must be the same length (per output channel)."""
    vals = np.asarray(vals)
    multipliers = np.asarray(multipliers)
    shifts = np.asarray(shifts)
    assert vals.shape == multipliers.shape == shifts.shape, (
        f"shape mismatch: vals={vals.shape} multipliers={multipliers.shape} shifts={shifts.shape}"
    )
    out = np.empty(vals.shape, dtype=np.int32)
    flat_out = out.reshape(-1)
    flat_vals = vals.reshape(-1)
    flat_mult = multipliers.reshape(-1)
    flat_shift = shifts.reshape(-1)
    for i in range(flat_vals.size):
        flat_out[i] = requantize(flat_vals[i], flat_mult[i], flat_shift[i])
    return out


def conv1d_int_reference(input_s8, weights_s8, bias_s32, input_offset,
                          output_offset, multipliers, shifts, activation_min, activation_max,
                          kernel_size, padding):
    """Bit-accurate int32-accumulation reference for a single Conv1D block
    laid out the way paci_infer.c maps it onto CMSIS-NN's NHWC Conv2D
    (H=1): input_s8 shape (W, C_IN), weights_s8 shape (C_OUT, K, C_IN),
    bias_s32 shape (C_OUT,). 'same' padding, stride 1, dilation 1 only
    (matches the frozen architecture, section 2 of the brief).

    Returns int8 output, shape (W, C_OUT), each channel requantized with
    its own (multiplier, shift), offset-added, and clamped to
    [activation_min, activation_max].
    """
    W, c_in = input_s8.shape
    c_out, k, c_in_w = weights_s8.shape
    assert k == kernel_size and c_in_w == c_in

    pad = (kernel_size - 1) // 2
    padded = np.zeros((W + 2 * pad, c_in), dtype=np.int32)
    padded[pad:pad + W, :] = input_s8.astype(np.int32) + input_offset

    out = np.zeros((W, c_out), dtype=np.int8)
    for w in range(W):
        window = padded[w:w + kernel_size, :]  # (K, C_IN)
        for co in range(c_out):
            acc = int(bias_s32[co])
            acc += int(np.sum(window.astype(np.int64) * weights_s8[co].astype(np.int64)))
            requant = int(requantize(acc, multipliers[co], shifts[co]))
            val = requant + output_offset
            val = max(activation_min, min(activation_max, val))
            out[w, co] = np.int8(val)
    return out


def global_average_pool_s8(input_s8, multiplier, shift, input_offset, output_offset,
                            activation_min, activation_max):
    """arm_avgpool_s8-equivalent reference over the whole window axis.
    input_s8 shape (W, C) -> output shape (C,)."""
    W, C = input_s8.shape
    out = np.zeros((C,), dtype=np.int8)
    for c in range(C):
        acc = int(np.sum(input_s8[:, c].astype(np.int64))) + input_offset * W
        # CMSIS-NN's avgpool divides by the window count via the same
        # multiplier/shift requantization path used elsewhere, not a plain
        # integer division, so the rounding behaviour matches conv/FC.
        requant = int(requantize(acc, multiplier, shift))
        val = requant + output_offset
        val = max(activation_min, min(activation_max, val))
        out[c] = np.int8(val)
    return out


def fully_connected_int_reference(input_s8, weights_s8, bias_s32, input_offset,
                                   output_offset, multipliers, shifts, activation_min, activation_max):
    """input_s8 shape (C_IN,), weights_s8 shape (C_OUT, C_IN), bias_s32
    shape (C_OUT,). Returns int8 output shape (C_OUT,)."""
    c_out, c_in = weights_s8.shape
    out = np.zeros((c_out,), dtype=np.int8)
    x = input_s8.astype(np.int64) + input_offset
    for co in range(c_out):
        acc = int(bias_s32[co]) + int(np.sum(x * weights_s8[co].astype(np.int64)))
        requant = int(requantize(acc, multipliers[co], shifts[co]))
        val = requant + output_offset
        val = max(activation_min, min(activation_max, val))
        out[co] = np.int8(val)
    return out


def make_tie_case(R, shift):
    """Construct (val, multiplier) such that requantize(val, multiplier, shift)
    lands on an EXACT tie: the pre-final-shift intermediate
    (new_val >> (total_shift - 1)) equals R exactly, R odd, so the discarded
    remainder at the function's last `(result + 1) >> 1` step is exactly
    one half — the case round-half-up and "round half to even"/"round half
    away from zero" disagree on. Returns None if R can't be represented in
    two int32 factors at this shift (only happens for R far outside a
    realistic accumulator's range).

    Construction: new_val = val * multiplier must equal exactly
    R << (total_shift - 1), with both val and multiplier fitting int32. The
    needed left-shift is split between the two factors — weighted toward
    `multiplier` first (up to 30 bits) — so `val` keeps as much headroom for
    R as possible even at extreme shift magnitudes.
    """
    total_shift = 31 - shift
    exponent = total_shift - 1
    if exponent < 0:
        return None

    b = min(exponent, 30)
    a = exponent - b
    if a > 30:
        return None

    val = R * (1 << a)
    multiplier = 1 << b
    if not (-(2**31) <= val <= 2**31 - 1):
        return None
    if not (0 <= multiplier <= 2**31 - 1):
        return None
    return val, multiplier
