#include "ekf.h"

void EKF_Init(EKF_t *ekf, float x0, float P0) {
    ekf->x = x0;
    ekf->P = P0;
    ekf->Q = EKF_Q_VAR;
    ekf->R = EKF_R_VAR;
}

float EKF_Step(EKF_t *ekf, float z, float pred_physics) {
    // 1. Predict
    // State transition f(x) = (1 - TAU)*x + TAU*pred_physics
    float x_pred = (1.0f - TAU) * ekf->x + TAU * pred_physics;
    
    // Jacobian F = (1 - TAU)
    float F = 1.0f - TAU;
    float P_pred = (F * ekf->P * F) + ekf->Q;
    
    // 2. Update
    // Measurement h(x) = x, so H = 1.0
    float innovation = z - x_pred;
    float S = P_pred + ekf->R;
    
    // Kalman Gain K = P_pred * H / S (where H=1)
    float K = P_pred / S;
    
    // State update
    ekf->x = x_pred + K * innovation;
    
    // Covariance update P = (1 - K*H) * P_pred
    ekf->P = (1.0f - K) * P_pred;
    
    // Calculate and return NIS (Normalized Innovation Squared)
    float nis = (innovation * innovation) / S;
    
    return nis;
}
