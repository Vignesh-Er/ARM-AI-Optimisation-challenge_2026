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

int8_t quantize_measurement(float z) {
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
    // ASSUMPTION: margin_threshold is derived from the rounding-bias budget
    // (Phase 5, section 5); 0 is a safe placeholder until then, not a
    // meaningful value. At 0, the margin-based escalation path below is
    // structurally wired up but never actually fires: paci_top2_margin
    // defines margin as (top logit - runner-up), so whenever Tier 1's
    // argmax class is 0 (normal), margin is >= 0 by construction, and
    // "margin < 0" can only be true when the top class is already
    // non-zero (which the class!=0 check already catches). Escalation
    // currently only happens via Tier-1-says-anomalous or watchdog.
    // Revisit once Phase 5 sets a real positive threshold, which will also
    // escalate low-confidence "normal" calls that this placeholder cannot.
    ctx->margin_threshold = 0;

    return PACI_OK;
}

// Tier 0 (EKF + NIS gate) runs in full below, followed by Tier 1/2
// dispatch per section 4: an NIS-triggered wake runs Tier 1 (the cheap
// INT4 screen) first, escalating to Tier 2 only if Tier 1 says anomalous
// or its margin is below ctx->margin_threshold (see the ASSUMPTION above
// paci_init — currently inert at the placeholder value 0). A watchdog-
// triggered wake goes straight to Tier 2, bypassing Tier 1 — the whole
// point of a forced periodic check is to catch what the coarse INT4
// screen's resolution might miss on slow drift, so re-running Tier 1
// first would be pointless. Both paths silently no-op (staying at
// PACI_TIER_0_EKF) if the ring isn't primed yet (paci_ring_read returns
// PACI_E_UNPRIMED) or the CMSIS-NN weights aren't built in
// (PACI_E_QUANT) — there is no window to classify yet in the first case,
// and nothing this function can do about the second.
paci_status_t paci_tier0_step(paci_ctx_t *ctx, float z,
                               float u_pressure, float u_temp,
                               float u_power, float u_flow,
                               float *out_nis) {
    if (ctx == NULL || out_nis == NULL) {
        return PACI_E_NULL;
    }
    float pred_physics = paci_physics_predict(u_pressure, u_temp, u_power, u_flow);
    paci_status_t status = paci_ekf_step(&ctx->ekf, z, pred_physics, out_nis);
    paci_ring_push(&ctx->ring, quantize_measurement(z));
    return status;
}

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

    float nis = 0.0f;
    paci_status_t ekf_status = paci_tier0_step(ctx, z, u_pressure, u_temp, u_power, u_flow, &nis);
    out->nis = nis;

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

    if (out->wake_reason == PACI_WAKE_NIS) {
        int8_t window[PACI_WINDOW_SIZE];
        if (paci_ring_read(&ctx->ring, window) == PACI_OK) {
            int8_t t1_class = -1;
            int32_t t1_margin = 0;
            if (paci_infer_t1_s4(window, &t1_class, &t1_margin) == PACI_OK) {
                ctx->n_t1++;
                out->tier_reached = PACI_TIER_1_INT4;
                out->class_id = t1_class;
                out->margin = t1_margin;

                bool escalate = (t1_class != 0) || (t1_margin < ctx->margin_threshold);
                if (escalate) {
                    int8_t t2_class = -1;
                    int32_t t2_margin = 0;
                    if (paci_infer_t2_s8(window, &t2_class, &t2_margin) == PACI_OK) {
                        ctx->n_t2++;
                        out->tier_reached = PACI_TIER_2_INT8;
                        out->class_id = t2_class;
                        out->margin = t2_margin;
                        if (t1_class == 0) {
                            // Escalated purely on a low margin, not because
                            // Tier 1 called it anomalous outright.
                            out->wake_reason = PACI_WAKE_MARGIN;
                            ctx->n_wake_margin++;
                        }
                    }
                }
            }
        }
    } else if (out->wake_reason == PACI_WAKE_WATCHDOG) {
        int8_t window[PACI_WINDOW_SIZE];
        if (paci_ring_read(&ctx->ring, window) == PACI_OK) {
            int8_t t2_class = -1;
            int32_t t2_margin = 0;
            if (paci_infer_t2_s8(window, &t2_class, &t2_margin) == PACI_OK) {
                ctx->n_t2++;
                out->tier_reached = PACI_TIER_2_INT8;
                out->class_id = t2_class;
                out->margin = t2_margin;
            }
        }
    }

    if (ekf_status != PACI_OK) {
        return ekf_status;
    }
    return PACI_OK;
}
