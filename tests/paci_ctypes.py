# SPDX-License-Identifier: Apache-2.0
"""ctypes bindings for paci_core, mirroring paci_core/include/paci_core.h and
paci_internal.h exactly (field order/types, so struct layout matches the C
compiler's natural alignment with no explicit packing on either side).

This is the single ctypes binding used by every test in tests/ — per G1,
there is exactly one EKF/physics/ring implementation (the C one); nothing
here reimplements any of that logic in Python.
"""
import ctypes
import os
import platform

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CANDIDATE_NAMES = ["libpaci_core.dll", "libpaci_core.so", "libpaci_core.dylib"]
_CANDIDATE_DIRS = [
    os.path.join(_PROJECT_ROOT, "build", "paci_core"),
    os.path.join(_PROJECT_ROOT, "build"),
]


def _find_shared_lib():
    for d in _CANDIDATE_DIRS:
        for name in _CANDIDATE_NAMES:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
    raise FileNotFoundError(
        "libpaci_core shared library not found. Build it first:\n"
        "  cmake -S . -B build && cmake --build build"
    )


PACI_WINDOW_SIZE = 32
PACI_N_CLASSES = 5

# paci_status_t
PACI_OK = 0
PACI_E_NULL = -1
PACI_E_UNPRIMED = -2
PACI_E_NUMERIC = -3
PACI_E_BUFSIZE = -4
PACI_E_QUANT = -5

# paci_tier_t
PACI_TIER_0_EKF = 0
PACI_TIER_1_INT4 = 1
PACI_TIER_2_INT8 = 2

# paci_wake_reason_t
PACI_WAKE_NONE = 0
PACI_WAKE_NIS = 1
PACI_WAKE_MARGIN = 2
PACI_WAKE_WATCHDOG = 3
PACI_WAKE_BURN_IN = 4


class PaciEkf(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("P", ctypes.c_float),
        ("Q", ctypes.c_float),
        ("R", ctypes.c_float),
        ("health_resets", ctypes.c_uint32),
    ]


class PaciRing(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_int8 * PACI_WINDOW_SIZE),
        ("head", ctypes.c_uint8),
        ("total", ctypes.c_uint32),
    ]


class PaciCtx(ctypes.Structure):
    _fields_ = [
        ("ekf", PaciEkf),
        ("ring", PaciRing),
        ("step_count", ctypes.c_uint32),
        ("steps_since_t2", ctypes.c_uint32),
        ("n_t1", ctypes.c_uint32),
        ("n_t2", ctypes.c_uint32),
        ("n_wake_nis", ctypes.c_uint32),
        ("n_wake_margin", ctypes.c_uint32),
        ("n_wake_watchdog", ctypes.c_uint32),
        ("margin_threshold", ctypes.c_int32),
    ]


class PaciResult(ctypes.Structure):
    _fields_ = [
        ("tier_reached", ctypes.c_int),
        ("wake_reason", ctypes.c_int),
        ("nis", ctypes.c_float),
        ("class_id", ctypes.c_int8),
        ("margin", ctypes.c_int32),
    ]


def load_lib():
    lib = ctypes.CDLL(_find_shared_lib())

    lib.paci_physics_predict.restype = ctypes.c_float
    lib.paci_physics_predict.argtypes = [
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]

    lib.paci_init.restype = ctypes.c_int
    lib.paci_init.argtypes = [ctypes.POINTER(PaciCtx), ctypes.c_float, ctypes.c_float]

    lib.paci_step.restype = ctypes.c_int
    lib.paci_step.argtypes = [
        ctypes.POINTER(PaciCtx),
        ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(PaciResult),
    ]

    lib.paci_ring_read.restype = ctypes.c_int
    lib.paci_ring_read.argtypes = [
        ctypes.POINTER(PaciRing), ctypes.POINTER(ctypes.c_int8),
    ]

    # Internal helpers (paci_internal.h) — real linkable symbols, used by
    # the D2/D3/G13 regression tests to drive the ring/EKF directly.
    lib.paci_ring_push.restype = None
    lib.paci_ring_push.argtypes = [ctypes.POINTER(PaciRing), ctypes.c_int8]

    lib.paci_ekf_step.restype = ctypes.c_int
    lib.paci_ekf_step.argtypes = [
        ctypes.POINTER(PaciEkf), ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]

    return lib


def ring_read(lib, ring):
    out = (ctypes.c_int8 * PACI_WINDOW_SIZE)()
    status = lib.paci_ring_read(ctypes.byref(ring), out)
    return status, list(out)
