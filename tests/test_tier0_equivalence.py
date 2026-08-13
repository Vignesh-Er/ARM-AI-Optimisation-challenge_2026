# SPDX-License-Identifier: Apache-2.0
import ctypes
import os
import sys
import copy

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from paci_ctypes import PACI_OK, PACI_WAKE_NONE, PACI_WAKE_NIS, PaciCtx, PaciResult, load_lib  # noqa: E402


@pytest.fixture(scope="module")
def lib():
    return load_lib()


def test_tier0_step_matches_monolith(lib):
    """
    Ensure that the extracted paci_tier0_step() behaves exactly the same
    as the monolith paci_step() up to the NIS check.
    """
    ctx1 = PaciCtx()
    ctx2 = PaciCtx()
    
    lib.paci_init(ctypes.byref(ctx1), config.ETCH_RATE_NOMINAL, config.P0_VAR)
    lib.paci_init(ctypes.byref(ctx2), config.ETCH_RATE_NOMINAL, config.P0_VAR)

    # We mock sensor inputs for 100 steps
    for k in range(100):
        # Generate some pseudo-random inputs
        v_sens1 = 0.5 + (k % 5) * 0.1
        v_sens2 = 0.5 + (k % 3) * 0.1
        v_sens3 = 0.5 + (k % 7) * 0.1
        v_sens4 = 0.5 + (k % 11) * 0.1

        v_sens5 = 0.5 + (k % 13) * 0.1

        # Evaluate monolith
        res1 = PaciResult()
        status1 = lib.paci_step(
            ctypes.byref(ctx1),
            v_sens1, v_sens2, v_sens3, v_sens4, v_sens5,
            ctypes.byref(res1)
        )

        # Evaluate decoupled tier0
        nis = ctypes.c_float()
        status2 = lib.paci_tier0_step(
            ctypes.byref(ctx2),
            v_sens1, v_sens2, v_sens3, v_sens4, v_sens5,
            ctypes.byref(nis)
        )

        # Check statuses match
        assert status1 == status2, f"Status mismatch at step {k}"
        
        # We only compare state if the monolith didn't escalate to Tier 1/2
        # because Tier 1/2 would modify state (like `n_t1`, `n_t2`, etc)
        # while paci_tier0_step obviously wouldn't.
        # But actually paci_tier0_step evaluates everything up to NIS.
        # EKF state and Ring buffer are updated identically in both.
        assert ctx1.ekf.x == pytest.approx(ctx2.ekf.x)
        assert ctx1.ekf.P == pytest.approx(ctx2.ekf.P)
        assert ctx1.ring.head == ctx2.ring.head
        assert ctx1.ekf.health_resets == ctx2.ekf.health_resets
        
        for i in range(config.WINDOW_SIZE):
            assert ctx1.ring.buf[i] == ctx2.ring.buf[i]
            
        # The NIS value should match exactly (if monolith woke on NIS, it's in res1.nis)
        # Actually res1.nis is always populated by the monolith
        if status1 == PACI_OK:
            assert res1.nis == pytest.approx(nis.value)
