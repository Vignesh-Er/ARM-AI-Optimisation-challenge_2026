// SPDX-License-Identifier: Apache-2.0
#ifndef PACI_CORE_H
#define PACI_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include "paci_params.h"   // generated from config.py — do not edit by hand

#ifdef __cplusplus
extern "C" {
#endif

#define PACI_WINDOW_SIZE   32
#define PACI_N_CLASSES     5

typedef enum {
    PACI_OK              =  0,
    PACI_E_NULL          = -1,   // null pointer argument
    PACI_E_UNPRIMED       = -2,   // fewer than PACI_WINDOW_SIZE samples seen
    PACI_E_NUMERIC       = -3,   // P or S non-positive; filter was reset
    PACI_E_BUFSIZE       = -4,   // CMSIS-NN scratch buffer too small
    PACI_E_QUANT         = -5    // exported quant params inconsistent
} paci_status_t;

typedef enum {
    PACI_TIER_0_EKF      = 0,
    PACI_TIER_1_INT4     = 1,
    PACI_TIER_2_INT8     = 2
} paci_tier_t;

typedef enum {
    PACI_WAKE_NONE       = 0,
    PACI_WAKE_NIS        = 1,   // NIS exceeded chi-square gate
    PACI_WAKE_MARGIN     = 2,   // Tier-1 margin below bias bound
    PACI_WAKE_WATCHDOG   = 3,   // forced periodic check
    PACI_WAKE_BURN_IN    = 4    // reserved; no inference during burn-in
} paci_wake_reason_t;

typedef struct {
    float    x;              // state estimate, etch rate (nm/min)
    float    P;              // state covariance, strictly > 0
    float    Q;              // process noise variance
    float    R;              // measurement noise variance
    uint32_t health_resets;  // times P or S went non-positive (G13)
} paci_ekf_t;

typedef struct {
    int8_t   buf[PACI_WINDOW_SIZE];  // int8-quantized samples, ring order
    uint8_t  head;                   // next write index, 0..31
    uint32_t total;                  // total samples ever written (never wraps
                                     // in practice; uint32_t is deliberate, D3)
} paci_ring_t;

typedef struct {
    paci_ekf_t  ekf;
    paci_ring_t ring;
    uint32_t    step_count;
    uint32_t    steps_since_t2;
    uint32_t    n_t1;                // Tier-1 invocations
    uint32_t    n_t2;                // Tier-2 invocations
    uint32_t    n_wake_nis;
    uint32_t    n_wake_margin;
    uint32_t    n_wake_watchdog;
    int32_t     margin_threshold;    // int8 logit units, from bias budget (§5)
} paci_ctx_t;

typedef struct {
    paci_tier_t         tier_reached;
    paci_wake_reason_t  wake_reason;
    float               nis;             // >= 0.0f; NAN never returned
    int8_t              class_id;        // 0..4 if tier_reached == TIER_2,
                                         // else -1
    int32_t             margin;          // top logit minus runner-up, int8
                                         // units; 0 if no inference ran
} paci_result_t;

// Deal-Grove-derived plasma etch rate. Units: p mTorr, T kelvin, w watts,
// f sccm; returns nm/min. Pure function, no globals, safe from an ISR.
float paci_physics_predict(float p, float T, float w, float f);

// Initialise. x0 in nm/min, P0 > 0. Returns PACI_E_NULL on null ctx,
// PACI_E_NUMERIC if P0 <= 0.
paci_status_t paci_init(paci_ctx_t *ctx, float x0, float P0);

// One cascade step. z = measurement (nm/min), the four u_* are the control
// inputs for the physics prediction. Writes *out. Never returns NAN in
// out->nis; on numeric trouble resets P, increments health_resets, and
// returns PACI_E_NUMERIC while still writing a valid *out.
// Not reentrant. Do not call from an ISR — see G9.
paci_status_t paci_step(paci_ctx_t *ctx,
                        float z,
                        float u_pressure, float u_temp,
                        float u_power,    float u_flow,
                        paci_result_t *out);

// Linearise the ring into chronological order, oldest first (D2).
// Returns PACI_E_UNPRIMED if ctx->ring.total < PACI_WINDOW_SIZE.
paci_status_t paci_ring_read(const paci_ring_t *r, int8_t out[PACI_WINDOW_SIZE]);

#ifdef __cplusplus
}
#endif
#endif // PACI_CORE_H
