# SPDX-License-Identifier: Apache-2.0
"""Task 2.3 (Oracle B foundation): tools/ref_cmsisnn.py's requantize() must
match the REAL compiled CMSIS-NN arm_nn_requantize exactly, not just be
internally self-consistent — Tier-1 (INT4) has no TFLite interpreter oracle
to check against, so this Python reference IS the oracle for it, and if the
reference itself were wrong, paci_infer_t1_s4() could pass its own check
while still being wrong relative to what real hardware runs.

An earlier version of ref_cmsisnn.requantize() used Python's
arbitrary-precision ints throughout, without emulating C's int32_t
truncation on the assignment `int32_t result = new_val >> (total_shift - 1);`
— that version mismatched the real compiled function on effectively all
large-shift inputs. This test is what caught that and must keep catching it.
"""
import ctypes
import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from tools.ref_cmsisnn import make_tie_case, requantize  # noqa: E402

_TIE_SHIFTS = list(range(-20, 20))
_TIES_PER_SHIFT = 100  # x2 after mirroring to negative R, per shift

_CANDIDATE_NAMES = ["librequantize_shim.dll", "librequantize_shim.so", "librequantize_shim.dylib"]
_CANDIDATE_DIRS = [
    os.path.join(_PROJECT_ROOT, "build"),
]


def _find_shim():
    for d in _CANDIDATE_DIRS:
        for name in _CANDIDATE_NAMES:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError(
        "librequantize_shim not found. Build it first:\n"
        "  cmake -S . -B build && cmake --build build"
    )


@pytest.fixture(scope="module")
def shim():
    lib = ctypes.CDLL(_find_shim())
    lib.paci_test_requantize.restype = ctypes.c_int32
    lib.paci_test_requantize.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    return lib


def test_requantize_matches_real_cmsisnn_over_random_inputs(shim):
    rng = np.random.RandomState(42)
    n_tested = 20000
    mismatches = []
    for _ in range(n_tested):
        val = int(rng.randint(-2**28, 2**28))
        multiplier = int(rng.randint(1, 2**31))
        shift = int(rng.randint(-20, 20))

        c_result = shim.paci_test_requantize(val, multiplier, shift)
        py_result = int(requantize(val, multiplier, shift))
        if c_result != py_result:
            mismatches.append((val, multiplier, shift, c_result, py_result))

    assert not mismatches, (
        f"{len(mismatches)}/{n_tested} mismatches between the real CMSIS-NN "
        f"arm_nn_requantize and tools/ref_cmsisnn.requantize(); first few: "
        f"{mismatches[:5]}"
    )


@pytest.mark.parametrize("val,multiplier,shift", [
    (0, 1, 0),
    (1, 1 << 30, 0),
    (-1, 1 << 30, 0),
    (2**27, 2**30, 15),
    (-(2**27), 2**30, 15),
    (2**20, 2**30 - 1, -10),
])
def test_requantize_matches_at_edge_cases(shim, val, multiplier, shift):
    c_result = shim.paci_test_requantize(val, multiplier, shift)
    py_result = int(requantize(val, multiplier, shift))
    assert c_result == py_result


def _positive_odd_r_values(shift, n, seed):
    """n distinct positive odd R values for which make_tie_case(R, shift)
    is representable in two int32 factors. max_r can be up to ~2^31 (at
    shift close to 31, where nearly all of the needed exponent lands on
    `multiplier` and `val` keeps almost the full int32 range) — sample
    directly rather than materialising range(1, max_r, 2), which would be a
    ~1-billion-element Python list for those shifts."""
    total_shift = 31 - shift
    exponent = total_shift - 1
    a = max(0, exponent - 30)
    max_r = (2**31 - 1) >> a  # largest R with R << a still fitting int32
    max_r = max(max_r, 1)
    max_odd_count = (max_r + 1) // 2

    rng = np.random.RandomState(seed)
    if max_odd_count <= n:
        return [2 * i + 1 for i in range(max_odd_count)]

    # rejection sampling into a set: O(n) expected regardless of how large
    # max_odd_count is, unlike rng.choice(huge_population, ..., replace=False)
    seen = set()
    while len(seen) < n:
        seen.add(int(rng.randint(0, max_odd_count)))
    return sorted(2 * i + 1 for i in seen)


@pytest.mark.parametrize("shift", _TIE_SHIFTS)
def test_exact_ties_match_compiled_function(shim, shift):
    """EXACT TIES: construct cases where the discarded remainder at the
    function's final round-half-up step is exactly one half (G8's anchor —
    this is precisely the case round-half-up and any zero-mean rounding
    scheme disagree on), for >= 200 cases per shift (100 positive R,
    mirrored to 100 negative R). Every case must match the real compiled
    arm_nn_requantize exactly."""
    r_values = _positive_odd_r_values(shift, _TIES_PER_SHIFT, seed=1000 + shift)
    assert len(r_values) >= 1, f"no representable tie R values at shift={shift}"

    mismatches = []
    tested = 0
    for R in r_values:
        for signed_r in (R, -R):
            case = make_tie_case(signed_r, shift)
            if case is None:
                continue
            val, multiplier = case
            tested += 1
            c_result = shim.paci_test_requantize(val, multiplier, shift)
            py_result = int(requantize(val, multiplier, shift))
            if c_result != py_result:
                mismatches.append((signed_r, val, multiplier, shift, c_result, py_result))

    assert tested >= min(2 * len(r_values), 2), f"constructed 0 usable tie cases at shift={shift}"
    assert not mismatches, f"{len(mismatches)}/{tested} tie mismatches at shift={shift}: {mismatches[:5]}"


@pytest.mark.parametrize("shift", [-20, -10, -1, 0, 1, 10, 19])
def test_tie_sign_asymmetry_matches_compiled_function(shim, shift):
    """SIGN ASYMMETRY: round-half-up rounds every tie toward positive
    infinity, not away from zero — so for odd R, result(R) + result(-R) == 1
    (NOT 0, which is what a magnitude-symmetric "round half away from zero"
    scheme would give). Verifies both the Python reference AND the real
    compiled function share this asymmetric behaviour, i.e. this isn't an
    artifact of the reference alone."""
    checked = 0
    for R in _positive_odd_r_values(shift, 20, seed=1000 + shift):
        pos_case = make_tie_case(R, shift)
        neg_case = make_tie_case(-R, shift)
        if pos_case is None or neg_case is None:
            continue
        checked += 1

        c_pos = shim.paci_test_requantize(*pos_case, shift)
        c_neg = shim.paci_test_requantize(*neg_case, shift)
        py_pos = int(requantize(*pos_case, shift))
        py_neg = int(requantize(*neg_case, shift))

        assert c_pos == py_pos and c_neg == py_neg
        assert c_pos + c_neg == 1, (
            f"R={R} shift={shift}: expected result(R) + result(-R) == 1 "
            f"(round-toward-+infinity signature), got {c_pos} + {c_neg} = {c_pos + c_neg}"
        )

    assert checked > 0, f"no usable sign-asymmetry cases constructed at shift={shift}"
