# SPDX-License-Identifier: Apache-2.0
"""D2/D3 regression tests: the original Core/Src/main_cascade.c wrote
sensor_buffer[buffer_index % WINDOW_SIZE] and passed sensor_buffer straight
to inference (rotated/time-scrambled unless buffer_index % 32 == 0), and
buffer_index was uint8_t (wraps at 256, silently skipping 31 windows every
256 steps instead of once at start-up). These tests fail against that old
behaviour and pass against paci_core/src/paci_ring.c.
"""
import ctypes

import pytest

from paci_ctypes import (
    PACI_E_UNPRIMED,
    PACI_OK,
    PaciCtx,
    PaciResult,
    PaciRing,
    load_lib,
    ring_read,
)


@pytest.fixture(scope="module")
def lib():
    return load_lib()


@pytest.mark.parametrize("phase", [0, 1, 5, 17, 31, 32, 63])
def test_ring_linearises_chronologically_at_arbitrary_phase(lib, phase):
    """D2: fill with a ramp 0..31 at an arbitrary phase; the linearised
    window must be exactly the ramp, oldest-first, regardless of where in
    the physical buffer the ramp happened to start."""
    ring = PaciRing()
    for _ in range(phase):
        lib.paci_ring_push(ctypes.byref(ring), ctypes.c_int8(-1))

    ramp = list(range(32))
    for v in ramp:
        lib.paci_ring_push(ctypes.byref(ring), ctypes.c_int8(v))

    status, window = ring_read(lib, ring)
    assert status == PACI_OK
    assert window == ramp
    assert all(window[i] < window[i + 1] for i in range(len(window) - 1))


def test_unprimed_count_over_1000_steps(lib):
    """D3: buffer_index must not be a uint8_t that wraps at 256. Over 1000
    steps, exactly the first 31 windows (total < 32) are unprimed."""
    ctx = PaciCtx()
    out = PaciResult()

    assert lib.paci_init(ctypes.byref(ctx), 250.0, 10.0) == PACI_OK

    skipped = 0
    for k in range(1000):
        z = 250.0 + 0.1 * k
        status = lib.paci_step(
            ctypes.byref(ctx), z, 50.0, 323.15, 500.0, 100.0, ctypes.byref(out)
        )
        assert status == PACI_OK
        read_status, _ = ring_read(lib, ctx.ring)
        if read_status == PACI_E_UNPRIMED:
            skipped += 1

    assert skipped == 31
