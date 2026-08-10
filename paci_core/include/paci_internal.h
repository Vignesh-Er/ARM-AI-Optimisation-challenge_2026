// SPDX-License-Identifier: Apache-2.0
#ifndef PACI_INTERNAL_H
#define PACI_INTERNAL_H

#include "paci_core.h"

#ifdef __cplusplus
extern "C" {
#endif

// Helpers shared between paci_core translation units. Not part of the public
// contract in paci_core.h (section 8 of the build brief) — callers outside
// this library should not include this header. It is still a real, linkable
// symbol boundary (not `static`) so tests/test_ring_buffer.py can drive
// paci_ring_push directly over ctypes without going through float
// quantization, matching the D2/D3 regression tests as specified.

// Push one already-quantized sample into the ring, advancing head/total.
void paci_ring_push(paci_ring_t *ring, int8_t sample);

// One EKF predict+update cycle. pred_physics is the physics-model output
// paci_physics_predict() already computed for this step. Writes *out_nis.
// Implements G13: if P_pred or S is non-positive, resets ekf->P to
// PACI_P0_VAR, increments ekf->health_resets, writes *out_nis = 0.0f (never
// NaN), and returns PACI_E_NUMERIC.
paci_status_t paci_ekf_step(paci_ekf_t *ekf, float z, float pred_physics, float *out_nis);

#ifdef __cplusplus
}
#endif
#endif // PACI_INTERNAL_H
