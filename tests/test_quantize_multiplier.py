# SPDX-License-Identifier: Apache-2.0
"""G7: tools/quantize_multiplier.py reimplements TFLite's QuantizeMultiplier
exactly, the canonical algorithm both the TFLite (Micro) interpreter and
CMSIS-NN's own TFLite integration use to derive (multiplier, shift) from a
tensor's stored float scale (the flatbuffer has no other representation to
read). Checked two ways: exact values for round scale factors (powers of
two, where the answer is unambiguous), and that requantize() using the
derived (multiplier, shift) approximates val * double_multiplier closely
for generic scales.
"""
import os
import random
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from tools.quantize_multiplier import quantize_multiplier  # noqa: E402
from tools.ref_cmsisnn import requantize  # noqa: E402


@pytest.mark.parametrize("double_multiplier,expected", [
    (0.0, (0, 0)),
    (0.5, (1 << 30, 0)),
    (1.0, (1 << 30, 1)),
    (0.25, (1 << 30, -1)),
    (2.0, (1 << 30, 2)),
])
def test_quantize_multiplier_exact_powers_of_two(double_multiplier, expected):
    assert quantize_multiplier(double_multiplier) == expected


def test_quantize_multiplier_output_is_valid_int32():
    rng = random.Random(0)
    for _ in range(5000):
        dm = rng.uniform(1e-8, 8.0)
        multiplier, shift = quantize_multiplier(dm)
        assert -(2**31) <= multiplier <= 2**31 - 1
        assert isinstance(shift, int)


def test_quantize_multiplier_reconstructs_scale_closely():
    """requantize(val, multiplier, shift) should approximate val*double_multiplier
    to within a small relative error for |val*double_multiplier| well above 1
    (small values are dominated by the +-0.5 rounding quantum, which is
    expected and not a bug)."""
    rng = random.Random(1)
    max_rel_err = 0.0
    for _ in range(3000):
        dm = rng.uniform(1e-6, 2.0)
        multiplier, shift = quantize_multiplier(dm)
        val = rng.randint(-2**20, 2**20)
        exact = val * dm
        actual = int(requantize(val, multiplier, shift))
        if abs(exact) > 100:  # well above the rounding quantum
            max_rel_err = max(max_rel_err, abs(actual - exact) / abs(exact))

    assert max_rel_err < 0.01, f"max relative error {max_rel_err} too large"
