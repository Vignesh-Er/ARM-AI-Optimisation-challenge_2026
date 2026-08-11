# SPDX-License-Identifier: Apache-2.0
"""Pack/unpack signed 4-bit values two-per-byte, in the exact order CMSIS-NN
expects. This is NOT derived from reading the DSP/MVE unpack intrinsics in
Include/arm_nnsupportfunctions.h (read_and_pad_s4 and neighbours) — that bit
-shuffle is for SIMD register loading and is not the wire format. The wire
format actually used to build CMSIS-NN's own validated test vectors is in
third_party/CMSIS-NN/Tests/UnitTest/conv_settings.py:299-300 (identical in
fully_connected_settings.py:212-213):

    temp = np.reshape(weights, (len(weights) // 2, 2)).astype(np.uint8)
    temp = 0xff & ((0xf0 & (temp[:, 1] << 4)) | (temp[:, 0] & 0xf))

i.e. for each consecutive pair (v0, v1) in the flat weight array: v0 (even
index) goes in the low nibble, v1 (odd index) goes in the high nibble. This
module reproduces that exactly and is reused by tools/export_cmsisnn.py.
"""
import numpy as np

S4_MIN = -8
S4_MAX = 7


def pack_s4(values):
    """Pack a flat array of signed int4 values (range [-8, 7]) two per byte,
    even index -> low nibble, odd index -> high nibble. Length must be even.
    Returns a uint8 numpy array of length len(values) // 2.
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"pack_s4 expects a flat 1-D array, got shape {values.shape}")
    if values.size % 2 != 0:
        raise ValueError(
            f"pack_s4: {values.size} elements is odd — CMSIS-NN's packed s4 "
            "format requires an even element count (G10). Pad the layer's "
            "channel/filter count to the nearest even value at export time; "
            "do not pack an odd-length array."
        )
    if values.size and (values.min() < S4_MIN or values.max() > S4_MAX):
        raise ValueError(
            f"pack_s4: values out of signed int4 range [{S4_MIN}, {S4_MAX}]: "
            f"min={values.min()}, max={values.max()}"
        )

    as_u8 = values.astype(np.int8).astype(np.uint8)
    pairs = as_u8.reshape(-1, 2)
    packed = (0xF0 & (pairs[:, 1].astype(np.uint16) << 4)) | (pairs[:, 0] & 0x0F)
    return (packed & 0xFF).astype(np.uint8)


def unpack_s4(packed, n):
    """Inverse of pack_s4. n is the number of signed int4 values to recover
    (must equal 2 * len(packed); passed explicitly so a truncated buffer is
    a loud error rather than a silently short result)."""
    packed = np.asarray(packed, dtype=np.uint8)
    if n != 2 * packed.size:
        raise ValueError(f"unpack_s4: n={n} does not match 2 * len(packed)={2 * packed.size}")

    low = (packed & 0x0F).astype(np.int16)
    high = ((packed >> 4) & 0x0F).astype(np.int16)
    # Sign-extend 4-bit two's complement (values 8..15 represent -8..-1).
    low = np.where(low >= 8, low - 16, low)
    high = np.where(high >= 8, high - 16, high)

    out = np.empty(n, dtype=np.int8)
    out[0::2] = low.astype(np.int8)
    out[1::2] = high.astype(np.int8)
    return out
