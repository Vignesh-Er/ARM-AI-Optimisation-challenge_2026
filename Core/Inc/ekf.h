#ifndef EKF_H
#define EKF_H

#ifdef __cplusplus
extern "C" {
#endif

// EKF Constants
#define EKF_Q_VAR 2.0f
#define EKF_R_VAR 4.0f
#define TAU 0.1f // Physics relaxation constant

typedef struct {
    float x;         // State estimate (etch rate)
    float P;         // Covariance
    float Q;         // Process noise covariance
    float R;         // Measurement noise covariance
    
    // Physics parameters for current step
    float nominal_rate;
} EKF_t;

// Initialize the EKF
void EKF_Init(EKF_t *ekf, float x0, float P0);

// Run one step of the EKF and return the NIS (Normalized Innovation Squared)
// z: sensor measurement
// pred_physics: the etch rate predicted purely by the physical Deal-Grove model
float EKF_Step(EKF_t *ekf, float z, float pred_physics);

#ifdef __cplusplus
}
#endif

#endif // EKF_H
