class ExtendedKalmanFilter:
    def __init__(self, x0, P0, Q, R, physics_model):
        """
        Args:
            x0: initial state estimate (scalar)
            P0: initial covariance (scalar)
            Q: process noise covariance (scalar)
            R: measurement noise covariance (scalar)  
            physics_model: PhysicsModel instance
        """
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self.model = physics_model
        
        self.nis = 0.0
        self.innovation = 0.0
        self.S = 0.0
        self.x_pred = x0
        self.P_pred = P0
    
    def predict(self, u):
        """EKF prediction step.
        x_pred = f(x, u)
        P_pred = F * P * F + Q
        """
        self.x_pred = self.model.state_transition(self.x, u)
        F = self.model.jacobian_F(self.x, u)
        self.P_pred = F * self.P * F + self.Q
    
    def update(self, z):
        """EKF update step.
        innovation y = z - h(x_pred)
        S = H * P_pred * H + R
        K = P_pred * H / S
        x = x_pred + K * y
        P = (1 - K*H) * P_pred
        NIS = y^2 / S
        
        Returns: (x_est, P, nis, innovation)
        """
        H = self.model.jacobian_H(self.x_pred)
        h_x = self.model.measurement_function(self.x_pred)
        
        self.innovation = z - h_x
        self.S = H * self.P_pred * H + self.R
        
        K = self.P_pred * H / self.S
        
        self.x = self.x_pred + K * self.innovation
        self.P = (1.0 - K * H) * self.P_pred
        
        self.nis = (self.innovation ** 2) / self.S
        return self.x, self.P, self.nis, self.innovation
    
    def step(self, u, z):
        """Combined predict + update.
        Returns: (x_est, P, nis, innovation)
        """
        self.predict(u)
        return self.update(z)
    
    def predict_only(self, u):
        """Predict without update (for when gate says SLEEP and we skip measurement update to protect state from corrupted readings)."""
        self.predict(u)
        self.x = self.x_pred
        self.P = self.P_pred
        return self.x, self.P
