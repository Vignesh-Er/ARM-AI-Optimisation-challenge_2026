import numpy as np

class IntelligentScheduler:
    def __init__(self, chi2_threshold=3.841, watchdog_interval=50,
                 adaptive_window=500, burn_in_steps=30):
        """
        Args:
            chi2_threshold: NIS threshold for triggering CNN (chi-square 95%)
            watchdog_interval: max cycles between forced CNN executions
            adaptive_window: rolling window size for adaptive Q/R estimation
            burn_in_steps: initial steps where gate is disabled (always SLEEP)
        """
        self.chi2_threshold = chi2_threshold
        self.watchdog_interval = watchdog_interval
        self.adaptive_window = adaptive_window
        self.burn_in_steps = burn_in_steps
        self.cycles_since_cnn = 0
        self.step_count = 0
        self._log = []
        self._recent_nis = []  # rolling buffer for adaptive estimation
    
    def step(self, nis):
        """Make a scheduling decision.
        
        Logic:
        1. During burn-in: always SLEEP (let EKF converge)
        2. After burn-in:
           a. If NIS > threshold → WAKE_CNN (anomaly detected)
           b. If cycles_since_cnn >= watchdog_interval → WAKE_CNN (forced)
           c. Otherwise → SLEEP
        
        On WAKE_CNN: reset cycles_since_cnn = 0
        On SLEEP: increment cycles_since_cnn
        
        Returns:
            tuple: (decision: str, reason: str)
            decision is 'WAKE_CNN' or 'SLEEP'
            reason is 'nis_exceeded', 'forced_watchdog', 'physics_sufficient', or 'burn_in'
        """
        self.step_count += 1
        self._update_adaptive_stats(nis)
        
        if self.step_count <= self.burn_in_steps:
            decision = 'SLEEP'
            reason = 'burn_in'
        else:
            if nis > self.chi2_threshold:
                decision = 'WAKE_CNN'
                reason = 'nis_exceeded'
            elif self.cycles_since_cnn >= self.watchdog_interval:
                decision = 'WAKE_CNN'
                reason = 'forced_watchdog'
            else:
                decision = 'SLEEP'
                reason = 'physics_sufficient'
                
        if decision == 'WAKE_CNN':
            self.cycles_since_cnn = 0
        else:
            self.cycles_since_cnn += 1
            
        self._log.append({'step': self.step_count, 'nis': nis, 'decision': decision, 'reason': reason})
        return decision, reason
    
    def _update_adaptive_stats(self, nis):
        """Track rolling NIS statistics for adaptive Q/R estimation.
        
        NOTE: Per analysis correction, we do NOT adapt the threshold itself
        (that would mask slow anomalies). Instead, we track stats that could
        be used to adapt Q/R covariance if needed.
        """
        self._recent_nis.append(nis)
        if len(self._recent_nis) > self.adaptive_window:
            self._recent_nis.pop(0)
    
    @property
    def log(self):
        """Return decision log as list of dicts."""
        return self._log
    
    @property
    def wake_rate(self):
        """Fraction of steps where CNN was invoked."""
        if not self._log:
            return 0.0
        wakes = sum(1 for entry in self._log if entry['decision'] == 'WAKE_CNN')
        return wakes / len(self._log)
    
    def get_stats(self):
        """Return summary statistics dict."""
        wakes = sum(1 for entry in self._log if entry['decision'] == 'WAKE_CNN')
        total = len(self._log)
        
        reasons = {}
        for entry in self._log:
            r = entry['reason']
            reasons[r] = reasons.get(r, 0) + 1
            
        return {
            'total_steps': total,
            'wake_count': wakes,
            'wake_rate': wakes / total if total > 0 else 0,
            'reasons': reasons
        }
