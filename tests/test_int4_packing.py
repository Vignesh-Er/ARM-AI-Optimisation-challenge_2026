# SPDX-License-Identifier: Apache-2.0
"""G10 regression test: packed int4 kernels pack two values per byte, so an
odd element count is either rejected or silently mis-strided. This tests
the pack/unpack round-trip (tools/int4_pack.py) against the full signed
int4 range and confirms an odd-length array raises a clear, loud error
instead of silently truncating or mis-packing.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.int4_pack import S4_MAX, S4_MIN, pack_s4, unpack_s4  # noqa: E402


def test_pack_unpack_round_trip_full_range():
    # Every possible signed int4 value, twice over, at an even length.
    values = np.array(list(range(S4_MIN, S4_MAX + 1)) * 2, dtype=np.int8)
    packed = pack_s4(values)
    assert packed.dtype == np.uint8
    assert len(packed) == len(values) // 2

    recovered = unpack_s4(packed, len(values))
    np.testing.assert_array_equal(recovered, values)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_pack_unpack_round_trip_random(seed):
    rng = np.random.RandomState(seed)
    n = rng.randint(2, 200) * 2  # always even
    values = rng.randint(S4_MIN, S4_MAX + 1, size=n).astype(np.int8)

    packed = pack_s4(values)
    recovered = unpack_s4(packed, n)
    np.testing.assert_array_equal(recovered, values)


def test_known_nibble_order():
    """Pins down the exact order verified against
    third_party/CMSIS-NN/Tests/UnitTest/conv_settings.py:299-300: even
    index -> low nibble, odd index -> high nibble."""
    values = np.array([3, -1], dtype=np.int8)  # v0=3 (low), v1=-1 (high)
    packed = pack_s4(values)
    assert len(packed) == 1
    # -1 as a 4-bit two's complement nibble is 0b1111 = 0xF; 3 is 0b0011.
    # Expected byte: (0xF << 4) | 0x3 = 0xF3.
    assert packed[0] == 0xF3


def test_odd_length_rejected_with_clear_error():
    with pytest.raises(ValueError, match="odd"):
        pack_s4(np.array([1, 2, 3], dtype=np.int8))


def test_out_of_range_rejected():
    with pytest.raises(ValueError, match="range"):
        pack_s4(np.array([8, -8], dtype=np.int8))  # 8 is out of [-8, 7]


def test_unpack_length_mismatch_rejected():
    packed = pack_s4(np.array([1, 2, 3, 4], dtype=np.int8))
    with pytest.raises(ValueError, match="does not match"):
        unpack_s4(packed, 3)
