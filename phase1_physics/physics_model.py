import numpy as np
from config import *

class PhysicsModel:
    def __init__(self):
        self.k0 = PHYSICS_K0
        self.alpha = PHYSICS_ALPHA
        self.beta = PHYSICS_BETA
        self.gamma = PHYSICS_GAMMA
        self.ea = PHYSICS_EA
        self.kb = KB_EV
        
        self.p_ref = P_REF
        self.t_ref = T_REF
        self.w_ref = W_REF
        self.f_ref = F_REF
        
        self.nominal_rate = ETCH_RATE_NOMINAL
        self.tau = TAU
        
    def etch_rate(self, u):
        """Compute physics-predicted etch rate from process parameters.
        
        Deal-Grove derived model:
        rate = ETCH_RATE_NOMINAL * (P/P_REF)^alpha * (F/F_REF)^beta * (W/W_REF)^gamma * exp(-EA/kB * (1/T - 1/T_REF))
        
        Args:
            u: array [pressure_mTorr, temperature_K, rf_power_W, gas_flow_sccm]
        Returns:
            predicted etch rate (nm/min)
        """
        P, T, W, F = u[0], u[1], u[2], u[3]
        term1 = (P / self.p_ref) ** self.alpha
        term2 = (F / self.f_ref) ** self.beta
        term3 = (W / self.w_ref) ** self.gamma
        term4 = np.exp(-self.ea / self.kb * (1.0 / T - 1.0 / self.t_ref))
        
        rate = self.nominal_rate * term1 * term2 * term3 * term4
        return rate
        
    def state_transition(self, x, u):
        """f(x, u) -> x_pred. State transition function.
        
        x_pred = (1 - TAU) * x + TAU * etch_rate(u)
        
        The actual etch rate relaxes toward the physics prediction
        with time constant TAU.
        
        Args:
            x: current etch rate (scalar float)
            u: process parameters array [P, T, W, F]
        Returns:
            predicted next etch rate (scalar float)
        """
        return (1.0 - self.tau) * x + self.tau * self.etch_rate(u)
        
    def jacobian_F(self, x, u):
        """Jacobian of f w.r.t. x.
        
        Since f(x,u) = (1-TAU)*x + TAU*etch_rate(u),
        df/dx = (1-TAU)
        
        Returns: scalar (1-TAU)
        """
        return 1.0 - self.tau
        
    def measurement_function(self, x):
        """h(x) = x. Identity measurement — sensor directly measures etch rate."""
        return x
        
    def jacobian_H(self, x):
        """Jacobian of h w.r.t. x = 1.0 (identity)"""
        return 1.0
