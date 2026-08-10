// SPDX-License-Identifier: Apache-2.0
#include "paci_internal.h"
#include <math.h>
#include <string.h>

// ASSUMPTION: the true CNN input quantization (scale/zero-point) is decided
// by the TFLite converter when phase4_tinyml/train.py quantizes the trained
// model, and tools/export_cmsisnn.py (Phase 2, G7) will extract those exact
// values from the flatbuffer. Until Phase 2 lands there is no trained model
// to extract them from, so the ring buffer here stores a provisional
// symmetric int8 quantization of the same normalized signal
// phase4_tinyml/dataset.py trains on: (measured - PACI_ETCH_RATE_NOMINAL) /
// PACI_NORM_SCALE. The scale below (0.25 per LSB) is sized so the hardest
// fault visible in the current dataset (Equipment Drift, ~30 in normalized
// units before the Phase 4 severity-ladder rework) does not saturate int8.
// Revisit at Phase 2: this #define must be replaced by the generated,
// TFLite-exported scale, and any ring contents quantized under it are
// provisional, not representative of the final input distribution.
#define PACI_PROVISIONAL_INPUT_SCALE 0.25f

static int8_t quantize_measurement(float z) {
    float norm = (z - PACI_ETCH_RATE_NOMINAL) / PACI_NORM_SCALE;
    float q = norm / PACI_PROVISIONAL_INPUT_SCALE;
    q = floorf(q + 0.5f);
    if (q > 127.0f) {
        q = 127.0f;
    }
    if (q < -128.0f) {
        q = -128.0f;
    }
    return (int8_t)q;
}

paci_status_t paci_init(paci_ctx_t *ctx, float x0, float P0) {
    if (ctx == NULL) {
        return PACI_E_NULL;
    }
    if (P0 <= 0.0f) {
        return PACI_E_NUMERIC;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->ekf.x = x0;
    ctx->ekf.P = P0;
    ctx->ekf.Q = PACI_Q_VAR;
    ctx->ekf.R = PACI_R_VAR;
    // margin_threshold is derived from the rounding-bias budget (Phase 5,
    // section 5) once Tier 1 exists; 0 until then is inert since
    // PACI_WAKE_MARGIN can't be reached before Phase 2 wires Tier 1.
    ctx->margin_threshold = 0;

    return PACI_OK;
}

// Tier 0 (EKF + NIS gate) runs in full below. Tier 1/2 dispatch — actually
// invoking paci_infer_t1_s4()/paci_infer_t2_s8() and advancing tier_reached
// past PACI_TIER_0_EKF — is added in Phase 2 once those CMSIS-NN kernels
// exist; see docs/STATUS.md. Until then this function still reports the
// *reason* Tier 1 would be woken (NIS gate, watchdog, burn-in), which is
// genuine Tier-0 evidence, not a stub of Tier-1/2 behaviour.
paci_status_t paci_step(paci_ctx_t *ctx,
                         float z,
                         float u_pressure, float u_temp,
                         float u_power,    float u_flow,
                         paci_result_t *out) {
    if (ctx == NULL || out == NULL) {
        return PACI_E_NULL;
    }

    memset(out, 0, sizeof(*out));
    out->class_id = -1;
    out->tier_reached = PACI_TIER_0_EKF;

    ctx->step_count++;

    float pred_physics = paci_physics_predict(u_pressure, u_temp, u_power, u_flow);

    float nis = 0.0f;
    paci_status_t ekf_status = paci_ekf_step(&ctx->ekf, z, pred_physics, &nis);
    out->nis = nis;

    paci_ring_push(&ctx->ring, quantize_measurement(z));

    if (ctx->step_count <= PACI_BURN_IN_STEPS) {
        out->wake_reason = PACI_WAKE_BURN_IN;
        ctx->steps_since_t2++;
    } else if (nis > PACI_CHI2_THRESHOLD_95) {
        out->wake_reason = PACI_WAKE_NIS;
        ctx->n_wake_nis++;
        ctx->steps_since_t2 = 0;
    } else if (ctx->steps_since_t2 >= PACI_WATCHDOG_INTERVAL) {
        out->wake_reason = PACI_WAKE_WATCHDOG;
        ctx->n_wake_watchdog++;
        ctx->steps_since_t2 = 0;
    } else {
        out->wake_reason = PACI_WAKE_NONE;
        ctx->steps_since_t2++;
    }

    if (ekf_status != PACI_OK) {
        return ekf_status;
    }
    return PACI_OK;
}
