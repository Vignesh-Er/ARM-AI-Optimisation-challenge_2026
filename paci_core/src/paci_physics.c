// SPDX-License-Identifier: Apache-2.0
#include "paci_core.h"
#include <math.h>

// Deal-Grove-derived plasma etch model, ported bit-for-bit (modulo float32
// vs float64) from phase1_physics/physics_model.py:PhysicsModel.etch_rate.
// Term order matches the Python source: pressure, gas flow, RF power, then
// the Arrhenius temperature term.
float paci_physics_predict(float p, float T, float w, float f) {
    float term_pressure  = powf(p / PACI_P_REF, PACI_PHYSICS_ALPHA);
    float term_flow      = powf(f / PACI_F_REF, PACI_PHYSICS_BETA);
    float term_power     = powf(w / PACI_W_REF, PACI_PHYSICS_GAMMA);
    float term_arrhenius = expf(-PACI_PHYSICS_EA / PACI_KB_EV * (1.0f / T - 1.0f / PACI_T_REF));

    return PACI_ETCH_RATE_NOMINAL * term_pressure * term_flow * term_power * term_arrhenius;
}
