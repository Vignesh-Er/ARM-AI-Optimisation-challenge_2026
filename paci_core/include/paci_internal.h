// SPDX-License-Identifier: Apache-2.0
#ifndef PACI_INTERNAL_H
#define PACI_INTERNAL_H

#include <stddef.h>
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

// Tier-1 (2-class, mixed INT8/INT4) and Tier-2 (5-class, INT8) CMSIS-NN
// inference, per section 5 of the Phase 2 continuation brief ("declared in
// the existing paci_core.h style" — kept here rather than literally inside
// paci_core.h, since section 8 mandates that file stay byte-for-byte exact
// and these two functions did not exist in the original contract).
// class_id is the argmax class; margin is top logit minus runner-up, int8/
// int32 logit units, computed on the RAW pre-softmax output (softmax is
// monotonic, so this changes neither the class nor the rank order the
// margin is measured over). Returns PACI_E_NULL on a null pointer,
// PACI_E_BUFSIZE if a layer's CMSIS-NN scratch requirement exceeds the
// static scratch buffer, PACI_E_QUANT if the generated weights header is
// missing (paci_core built without running tools/export_cmsisnn.py first)
// or a CMSIS-NN kernel call itself reports failure.
paci_status_t paci_infer_t1_s4(const int8_t window[PACI_WINDOW_SIZE], int8_t *class_id, int32_t *margin);
paci_status_t paci_infer_t2_s8(const int8_t window[PACI_WINDOW_SIZE], int8_t *class_id, int32_t *margin);
int8_t quantize_measurement(float raw_z);
paci_status_t paci_tier0_step(paci_ctx_t *ctx, float z,
                               float u_pressure, float u_temp,
                               float u_power, float u_flow,
                               float *out_nis);

#ifdef __cplusplus
}
#endif
#endif // PACI_INTERNAL_H
