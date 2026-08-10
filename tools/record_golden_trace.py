# SPDX-License-Identifier: Apache-2.0
"""Record a golden trace of paci_core's C EKF (via ctypes) over the standard
2000-step / seed-42 dataset, for tests/test_bitexact.py (G1) to check future
changes against. Only run this deliberately, on purpose, when the C core's
numeric behaviour has intentionally changed — it overwrites the committed
golden trace that regression-tests everyone else's changes.

Usage: python tools/record_golden_trace.py
"""
import ctypes
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tests"))

import config  # noqa: E402
from phase1_physics.physics_model import PhysicsModel  # noqa: E402
from phase1_physics.synthetic_data import generate_full_dataset  # noqa: E402

from paci_ctypes import PaciCtx, PaciResult, load_lib  # noqa: E402

_GOLDEN_PATH = os.path.join(_PROJECT_ROOT, "tests", "golden", "ekf_trace_seed42.json")


def run_trace():
    lib = load_lib()
    physics = PhysicsModel()
    dataset = generate_full_dataset(physics, n_steps=config.N_STEPS, seed=config.SEED)

    ctx = PaciCtx()
    status = lib.paci_init(ctypes.byref(ctx), config.ETCH_RATE_NOMINAL, config.P0_VAR)
    assert status == 0, f"paci_init failed: {status}"

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


def main():
    trace = run_trace()
    os.makedirs(os.path.dirname(_GOLDEN_PATH), exist_ok=True)
    with open(_GOLDEN_PATH, "w", newline="\n", encoding="ascii") as f:
        json.dump(trace, f, indent=1)
    print(f"Wrote {len(trace)}-step golden trace to {_GOLDEN_PATH}")


if __name__ == "__main__":
    main()
