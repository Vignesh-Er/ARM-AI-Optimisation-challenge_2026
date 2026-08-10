# SPDX-License-Identifier: Apache-2.0
"""G13 regression tests: the non-Joseph covariance update
P = (1 - K)*P_pred can, with a bad Q/R, drive P (and S = P_pred + R)
non-positive; NIS then goes negative or NaN and the gate silently stops
firing. paci_ekf_step must detect this, reset P to config.P0_VAR, count it
in health_resets, and report NIS as 0.0 (never NaN).
"""
import ctypes
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

from paci_ctypes import PACI_E_NUMERIC, PACI_OK, PaciEkf, load_lib  # noqa: E402


@pytest.fixture(scope="module")
def lib():
    return load_lib()


def test_ekf_resets_on_non_positive_S(lib):
    # R this negative forces S = P_pred + R <= 0 on the very first call.
    ekf = PaciEkf(x=config.ETCH_RATE_NOMINAL, P=config.P0_VAR, Q=config.Q_VAR, R=-100.0, health_resets=0)
    nis = ctypes.c_float(-1.0)

    status = lib.paci_ekf_step(
        ctypes.byref(ekf), config.ETCH_RATE_NOMINAL, config.ETCH_RATE_NOMINAL, ctypes.byref(nis)
    )

    assert status == PACI_E_NUMERIC
    assert ekf.health_resets == 1
    assert ekf.P == pytest.approx(config.P0_VAR)
    assert nis.value == 0.0


def test_ekf_reset_is_cumulative_across_repeated_faults(lib):
    ekf = PaciEkf(x=config.ETCH_RATE_NOMINAL, P=config.P0_VAR, Q=config.Q_VAR, R=-5.0, health_resets=0)
    nis = ctypes.c_float()

    for _ in range(3):
        status = lib.paci_ekf_step(
            ctypes.byref(ekf), config.ETCH_RATE_NOMINAL, config.ETCH_RATE_NOMINAL, ctypes.byref(nis)
        )
        assert status == PACI_E_NUMERIC

    assert ekf.health_resets == 3


def test_ekf_healthy_operation_never_resets_or_produces_nan(lib):
    ekf = PaciEkf(x=config.ETCH_RATE_NOMINAL, P=config.P0_VAR, Q=config.Q_VAR, R=config.R_VAR, health_resets=0)
    nis = ctypes.c_float()

    for k in range(2000):
        z = config.ETCH_RATE_NOMINAL + 2.0 * ((k % 7) - 3)  # bounded, deterministic
        status = lib.paci_ekf_step(
            ctypes.byref(ekf), z, config.ETCH_RATE_NOMINAL, ctypes.byref(nis)
        )
        assert status == PACI_OK
        assert ekf.health_resets == 0
        assert not math.isnan(nis.value)
        assert nis.value >= 0.0
