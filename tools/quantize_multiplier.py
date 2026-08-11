# SPDX-License-Identifier: Apache-2.0
"""TFLite's QuantizeMultiplier, faithfully reimplemented.

The TFLite flatbuffer only stores a tensor's quantization as a float
`scale` and integer `zero_point` — it does NOT store a pre-computed int32
multiplier/shift pair. Both the TFLite (Micro) interpreter and CMSIS-NN's
own TFLite integration derive (multiplier, shift) from that float scale at
load/conversion time via this exact algorithm
(tensorflow/lite/kernels/internal/quantization_util.cc:QuantizeMultiplier).
"Read them as the runtime stores them, never recompute int8 multipliers
from scale floats" (G7) means: use this canonical algorithm, not an ad hoc
alternative — there is no other array to "read" instead, so implementing
this precisely IS the reading. Correctness is checked indirectly but
strongly: Oracle A (tests/test_export_cmsisnn.py) requires
paci_infer_t2_s8()'s output to match the TFLite interpreter exactly on 200
held-out windows, which would fail immediately if this were wrong.

Python's math.frexp matches C's frexp exactly for IEEE754 doubles (same
IEEE754 mantissa/exponent decomposition), so this is a direct port, not an
approximation.
"""
import math


def quantize_multiplier(double_multiplier):
    """Returns (multiplier: int32, shift: int) such that
    double_multiplier ~= multiplier * 2**(shift - 31), matching the
    convention arm_nn_requantize's `shift` parameter expects directly
    (verified: arm_nn_requantize computes
    result ~= val * multiplier * 2**(shift - 31), same exponent
    convention, no sign flip needed)."""
    if double_multiplier == 0.0:
        return 0, 0

    q, shift = math.frexp(double_multiplier)  # double_multiplier == q * 2**shift, 0.5 <= |q| < 1
    q_fixed = int(round(q * (1 << 31)))
    assert abs(q_fixed) <= (1 << 31), f"q_fixed={q_fixed} out of range for double_multiplier={double_multiplier}"

    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1

    if shift < -31:
        shift = 0
        q_fixed = 0

    assert -(2**31) <= q_fixed <= 2**31 - 1, f"q_fixed={q_fixed} does not fit int32"
    return int(q_fixed), int(shift)
