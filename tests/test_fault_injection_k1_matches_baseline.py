# SPDX-License-Identifier: Apache-2.0
"""Task 2.0 regression test: adding the severity multiplier k to
inject_fault() must not change behaviour at k=1.0 for gas_leak,
equipment_drift, and unexpected_deviation (verified byte-for-byte against
tests/golden/synthetic_data_baseline.npz, captured from the pre-k code).

sensor_fault is the one deliberate exception: it was redefined from
flatline-to-0.0 (a ~125-sigma jump) to stuck-at-last-reading, per the
project owner's instruction, so it is NOT expected to match the baseline —
this test instead asserts the new k=1.0 behaviour is what was asked for
(a hard stick: constant, zero-variance, equal to the pre-fault reading).
"""
import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from phase1_physics.physics_model import PhysicsModel  # noqa: E402
from phase1_physics.synthetic_data import generate_full_dataset  # noqa: E402

_BASELINE_PATH = os.path.join(_PROJECT_ROOT, "tests", "golden", "synthetic_data_baseline.npz")


def _fault_segments(labels):
    """Return {label: (start, end)} for each contiguous nonzero label run."""
    segments = {}
    prev = 0
    start = None
    for i, label in enumerate(labels):
        if label != prev:
            if prev != 0:
                segments[int(prev)] = (start, i)
            if label != 0:
                start = i
            prev = label
    if prev != 0:
        segments[int(prev)] = (start, len(labels))
    return segments


@pytest.fixture(scope="module")
def baseline():
    return np.load(_BASELINE_PATH)


@pytest.fixture(scope="module")
def live_k1():
    physics = PhysicsModel()
    return generate_full_dataset(physics, n_steps=config.N_STEPS, seed=config.SEED, k=1.0)


def test_labels_unchanged(baseline, live_k1):
    assert np.array_equal(baseline["labels"], live_k1["labels"])


@pytest.mark.parametrize("label,name", [(2, "gas_leak"), (3, "equipment_drift"), (4, "unexpected_deviation")])
def test_k1_matches_baseline_byte_for_byte(baseline, live_k1, label, name):
    segments = _fault_segments(baseline["labels"])
    start, end = segments[label]
    np.testing.assert_array_equal(
        baseline["measured_etch_rate"][start:end], live_k1["measured_etch_rate"][start:end],
        err_msg=f"{name} (label {label}) diverged from baseline at k=1.0",
    )
    np.testing.assert_array_equal(
        baseline["true_etch_rate"][start:end], live_k1["true_etch_rate"][start:end],
        err_msg=f"{name} (label {label}) true_etch_rate diverged from baseline at k=1.0",
    )


def test_sensor_fault_redefinition_is_hard_stick_at_k1(baseline, live_k1):
    """Documents the one intentional exception: sensor_fault at k=1.0 does
    NOT match the old flatline-to-0.0 baseline — it's a hard stick instead."""
    segments = _fault_segments(baseline["labels"])
    start, end = segments[1]

    old_segment = baseline["measured_etch_rate"][start:end]
    new_segment = live_k1["measured_etch_rate"][start:end]

    assert not np.array_equal(old_segment, new_segment), (
        "sensor_fault should differ from the old flatline-to-0.0 baseline — "
        "if this now matches, the redefinition was accidentally reverted"
    )
    assert np.all(old_segment == 0.0), "sanity check: baseline really was flatline-to-0.0"

    # Hard stick: constant, equal to the reading immediately before fault onset.
    # (Not also asserting np.var(...) == 0.0: np.var's mean-then-sum-of-squares
    # path picks up ~1e-26 float64 rounding noise even over bit-identical
    # elements, which the direct element-wise equality below already proves.)
    stuck_value = live_k1["measured_etch_rate"][start - 1]
    assert np.all(new_segment == stuck_value)
