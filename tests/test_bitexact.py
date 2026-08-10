# SPDX-License-Identifier: Apache-2.0
"""G1 regression test: Python float64 and C float32 will drift, so the C
core is the single reference implementation — Python calls into it via
ctypes rather than maintaining a second EKF. This test re-runs paci_core
over the standard 2000-step/seed-42 dataset and checks the live ctypes run
against a recorded golden trace (tests/golden/ekf_trace_seed42.json,
produced by tools/record_golden_trace.py) to within 1 ULP of float32.

Tolerance: the live run and the golden trace both call the exact same
compiled paci_core binary, so bit-identical (0 ULP) output is the norm; the
1-ULP margin (via numpy.spacing at each recorded value, not a fixed epsilon)
only guards against innocuous non-determinism such as denormal-handling
differences across the machine that recorded the golden trace and the one
running CI, per the brief's explicit "within 1 ULP" requirement — it is not
there to paper over a real divergence.
"""
import ctypes
import json
import math
import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from phase1_physics.physics_model import PhysicsModel  # noqa: E402
from phase1_physics.synthetic_data import generate_full_dataset  # noqa: E402

from paci_ctypes import PaciCtx, PaciResult, load_lib  # noqa: E402

_GOLDEN_PATH = os.path.join(_PROJECT_ROOT, "tests", "golden", "ekf_trace_seed42.json")


def _within_one_ulp(a, b):
    a32 = np.float32(a)
    b32 = np.float32(b)
    if a32 == b32:
        return True
    ulp = float(np.spacing(a32)) or float(np.spacing(np.float32(1.0)))
    return abs(float(a32) - float(b32)) <= ulp


@pytest.fixture(scope="module")
def golden_trace():
    with open(_GOLDEN_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def live_trace():
    lib = load_lib()
    physics = PhysicsModel()
    dataset = generate_full_dataset(physics, n_steps=config.N_STEPS, seed=config.SEED)

    ctx = PaciCtx()
    assert lib.paci_init(ctypes.byref(ctx), config.ETCH_RATE_NOMINAL, config.P0_VAR) == 0

    out = PaciResult()
    trace = []
    for k in range(config.N_STEPS):
        u_pressure = float(dataset["params"]["pressure"][k])
        u_temp = float(dataset["params"]["temperature"][k])
        u_power = float(dataset["params"]["rf_power"][k])
        u_flow = float(dataset["params"]["gas_flow"][k])
        z = float(dataset["measured_etch_rate"][k])

        status = lib.paci_step(
            ctypes.byref(ctx), z, u_pressure, u_temp, u_power, u_flow, ctypes.byref(out)
        )
        trace.append({
            "step": k,
            "status": status,
            "x": ctx.ekf.x,
            "P": ctx.ekf.P,
            "nis": out.nis,
            "wake_reason": out.wake_reason,
            "health_resets": ctx.ekf.health_resets,
        })
    return trace


def test_golden_trace_exists():
    assert os.path.isfile(_GOLDEN_PATH), (
        "No golden trace recorded. Run `python tools/record_golden_trace.py` "
        "once and commit tests/golden/ekf_trace_seed42.json."
    )


def test_live_run_matches_golden_trace_within_one_ulp(golden_trace, live_trace):
    assert len(live_trace) == len(golden_trace) == config.N_STEPS

    for expected, actual in zip(golden_trace, live_trace):
        step = expected["step"]
        assert actual["status"] == expected["status"], f"step {step}: status mismatch"
        assert actual["wake_reason"] == expected["wake_reason"], f"step {step}: wake_reason mismatch"
        assert actual["health_resets"] == expected["health_resets"], f"step {step}: health_resets mismatch"
        assert not math.isnan(actual["x"]), f"step {step}: x is NaN"
        assert not math.isnan(actual["P"]), f"step {step}: P is NaN"
        assert not math.isnan(actual["nis"]), f"step {step}: nis is NaN"
        assert _within_one_ulp(actual["x"], expected["x"]), f"step {step}: x diverged beyond 1 ULP"
        assert _within_one_ulp(actual["P"], expected["P"]), f"step {step}: P diverged beyond 1 ULP"
        assert _within_one_ulp(actual["nis"], expected["nis"]), f"step {step}: nis diverged beyond 1 ULP"
