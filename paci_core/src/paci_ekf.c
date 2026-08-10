// SPDX-License-Identifier: Apache-2.0
#include "paci_internal.h"

// Scalar EKF predict+update, ported from phase2_ekf/ekf.py:ExtendedKalmanFilter
// (predict() + update()), using PACI_TAU as the state-transition relaxation
// factor in place of a full physics-model Jacobian call (H = 1, identity
// measurement, matches the Python model's jacobian_H/measurement_function).
//
// G13: the non-Joseph covariance update P = (1-K)*P_pred can, with a bad Q/R,
// drive P (and therefore S = P_pred + R) non-positive, after which NIS goes
// negative or NaN and the gate silently stops firing. Guard both P_pred and S
// before they're used in a division, and guard P again after the update.
paci_status_t paci_ekf_step(paci_ekf_t *ekf, float z, float pred_physics, float *out_nis) {
    float F = 1.0f - PACI_TAU;
    float x_pred = F * ekf->x + PACI_TAU * pred_physics;
    float P_pred = F * ekf->P * F + ekf->Q;

    float innovation = z - x_pred;
    float S = P_pred + ekf->R;

    if (P_pred <= 0.0f || S <= 0.0f) {
        ekf->x = x_pred;
        ekf->P = PACI_P0_VAR;
        ekf->health_resets++;
        *out_nis = 0.0f;
        return PACI_E_NUMERIC;
    }

    float K = P_pred / S;
    ekf->x = x_pred + K * innovation;
    ekf->P = (1.0f - K) * P_pred;

    if (ekf->P <= 0.0f) {
        ekf->P = PACI_P0_VAR;
        ekf->health_resets++;
        *out_nis = 0.0f;
        return PACI_E_NUMERIC;
    }

    *out_nis = (innovation * innovation) / S;
    return PACI_OK;
}
