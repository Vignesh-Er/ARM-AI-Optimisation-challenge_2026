"""
Phase 6 — Baseline Methods

Implements all baseline gating strategies for comparison against PACI:
1. Always-run CNN (no gating)
2. Variance threshold gate
3. Moving average gate
4. CUSUM detector gate
5. Kalman innovation gate (no physics model — random walk)
"""
import numpy as np


class AlwaysOnBaseline:
    """No gating — CNN runs every step."""
    name = "Always-On CNN"
    
    def step(self, measurement):
        return 'WAKE_CNN', 'always_on'
    
    @property
    def wake_rate(self):
        return 1.0


class VarianceThresholdGate:
    """Statistical gate using rolling variance of raw measurements."""
    name = "Variance Threshold"
    
    def __init__(self, window=50, threshold_multiplier=3.0, burn_in=30):
        self.window = window
        self.threshold_multiplier = threshold_multiplier
        self.burn_in = burn_in
        self.buffer = []
        self.baseline_var = None
        self.step_count = 0
        self._wake_count = 0
    
    def step(self, measurement):
        self.step_count += 1
        self.buffer.append(measurement)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        
        if self.step_count <= self.burn_in:
            return 'SLEEP', 'burn_in'
        
        if self.step_count == self.burn_in + 1:
            self.baseline_var = np.var(self.buffer)
        
        if self.baseline_var is not None and self.baseline_var > 0:
            current_var = np.var(self.buffer[-min(20, len(self.buffer)):])
            if current_var > self.threshold_multiplier * self.baseline_var:
                self._wake_count += 1
                return 'WAKE_CNN', 'variance_exceeded'
        
        return 'SLEEP', 'normal_variance'
    
    @property
    def wake_rate(self):
        return self._wake_count / self.step_count if self.step_count > 0 else 0


class MovingAverageGate:
    """Gate based on deviation from a moving average."""
    name = "Moving Average"
    
    def __init__(self, window=50, threshold_sigma=3.0, burn_in=30):
        self.window = window
        self.threshold_sigma = threshold_sigma
        self.burn_in = burn_in
        self.buffer = []
        self.step_count = 0
        self._wake_count = 0
    
    def step(self, measurement):
        self.step_count += 1
        self.buffer.append(measurement)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        
        if self.step_count <= self.burn_in:
            return 'SLEEP', 'burn_in'
        
        ma = np.mean(self.buffer)
        std = np.std(self.buffer) if len(self.buffer) > 1 else 1.0
        
        if std > 0 and abs(measurement - ma) > self.threshold_sigma * std:
            self._wake_count += 1
            return 'WAKE_CNN', 'deviation_exceeded'
        
        return 'SLEEP', 'within_range'
    
    @property
    def wake_rate(self):
        return self._wake_count / self.step_count if self.step_count > 0 else 0


class CUSUMGate:
    """Cumulative Sum (CUSUM) change-point detector."""
    name = "CUSUM Detector"
    
    def __init__(self, threshold=5.0, drift=0.5, burn_in=30):
        self.threshold = threshold
        self.drift = drift
        self.burn_in = burn_in
        self.S_pos = 0.0
        self.S_neg = 0.0
        self.mean = None
        self.buffer = []
        self.step_count = 0
        self._wake_count = 0
    
    def step(self, measurement):
        self.step_count += 1
        self.buffer.append(measurement)
        
        if self.step_count <= self.burn_in:
            return 'SLEEP', 'burn_in'
        
        if self.mean is None:
            self.mean = np.mean(self.buffer)
            self.std = np.std(self.buffer) if np.std(self.buffer) > 0 else 1.0
        
        z = (measurement - self.mean) / self.std
        
        self.S_pos = max(0, self.S_pos + z - self.drift)
        self.S_neg = max(0, self.S_neg - z - self.drift)
        
        if self.S_pos > self.threshold or self.S_neg > self.threshold:
            self._wake_count += 1
            self.S_pos = 0.0
            self.S_neg = 0.0
            return 'WAKE_CNN', 'cusum_alarm'
        
        return 'SLEEP', 'no_alarm'
    
    @property
    def wake_rate(self):
        return self._wake_count / self.step_count if self.step_count > 0 else 0


class KalmanNoPhysicsGate:
    """Kalman filter with random-walk process model (no physics prior).
    
    This is the key ablation: same NIS gating idea, but without the
    physics model — the process model is x[k+1] = x[k] (random walk).
    """
    name = "Kalman (No Physics)"
    
    def __init__(self, chi2_threshold=3.841, Q=2.0, R=4.0, burn_in=30, watchdog=50):
        self.threshold = chi2_threshold
        self.x = 250.0  # initial guess
        self.P = 10.0
        self.Q = Q
        self.R = R
        self.burn_in = burn_in
        self.watchdog = watchdog
        self.step_count = 0
        self.cycles_since_cnn = 0
        self._wake_count = 0
    
    def step(self, measurement):
        self.step_count += 1
        
        # Predict (random walk: x_pred = x, P_pred = P + Q)
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update
        innovation = measurement - x_pred
        S = P_pred + self.R
        K = P_pred / S
        self.x = x_pred + K * innovation
        self.P = (1 - K) * P_pred
        
        nis = (innovation ** 2) / S
        
        if self.step_count <= self.burn_in:
            return 'SLEEP', 'burn_in'
        
        if nis > self.threshold or self.cycles_since_cnn >= self.watchdog:
            self._wake_count += 1
            self.cycles_since_cnn = 0
            reason = 'nis_exceeded' if nis > self.threshold else 'forced_watchdog'
            return 'WAKE_CNN', reason
        
        self.cycles_since_cnn += 1
        return 'SLEEP', 'within_threshold'
    
    @property
    def wake_rate(self):
        return self._wake_count / self.step_count if self.step_count > 0 else 0


# Registry of all baselines for easy iteration
ALL_BASELINES = [
    AlwaysOnBaseline,
    VarianceThresholdGate,
    MovingAverageGate,
    CUSUMGate,
    KalmanNoPhysicsGate,
]
