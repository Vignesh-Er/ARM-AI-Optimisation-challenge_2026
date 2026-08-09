import os
import numpy as np
import matplotlib.pyplot as plt
from config import *
from phase1_physics.physics_model import PhysicsModel
from phase1_physics.synthetic_data import generate_full_dataset
from phase2_ekf.ekf import ExtendedKalmanFilter

def main():
    model = PhysicsModel()
    data = generate_full_dataset(model, N_STEPS, SEED)
    
    ekf = ExtendedKalmanFilter(
        x0=ETCH_RATE_NOMINAL,
        P0=P0_VAR,
        Q=Q_VAR,
        R=R_VAR,
        physics_model=model
    )
    
    n_steps = len(data['labels'])
    x_est = np.zeros(n_steps)
    P_est = np.zeros(n_steps)
    nis = np.zeros(n_steps)
    innovation = np.zeros(n_steps)
    
    for i in range(n_steps):
        u = np.array([data['params']['pressure'][i], data['params']['temperature'][i], 
                      data['params']['rf_power'][i], data['params']['gas_flow'][i]])
        z = data['measured_etch_rate'][i]
        
        x_est[i], P_est[i], nis[i], innovation[i] = ekf.step(u, z)
        
    # Validation Plots
    # 1. State estimate
    plt.figure(figsize=(10, 5))
    plt.plot(data['true_etch_rate'], label='True Etch Rate')
    plt.plot(x_est, label='EKF Estimate', alpha=0.7)
    plt.ylabel('Etch Rate')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase2_ekf_state_estimate.png'))
    plt.close()
    
    # 2. Innovation
    plt.figure(figsize=(10, 5))
    plt.plot(innovation, label='Innovation')
    bound = 2 * np.sqrt(P_est + R_VAR)
    plt.plot(bound, 'r--', label='+2 sigma')
    plt.plot(-bound, 'r--', label='-2 sigma')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase2_ekf_innovation.png'))
    plt.close()
    
    # 3. Covariance
    plt.figure(figsize=(10, 5))
    plt.plot(P_est, label='P (Covariance)')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase2_ekf_covariance.png'))
    plt.close()
    
    # 4. NIS
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(nis, label='NIS')
    plt.axhline(CHI2_THRESHOLD_95, color='r', linestyle='--', label='95% Threshold')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    normal_nis = nis[data['labels'] == 0]
    plt.hist(normal_nis, bins=30, density=True, label='Normal NIS')
    from scipy.stats import chi2
    x_val = np.linspace(0, max(normal_nis), 100)
    plt.plot(x_val, chi2.pdf(x_val, 1), 'r-', label='Chi2(1) PDF')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase2_ekf_nis.png'))
    plt.close()
    
    # Chi-square consistency test
    pass_count = np.sum(normal_nis < CHI2_THRESHOLD_95)
    pass_rate = pass_count / len(normal_nis)
    test_result = "PASS" if abs(pass_rate - 0.95) < 0.05 else "FAIL"
    print(f"Chi-square test (Normal data): {pass_rate*100:.2f}% < threshold. Result: {test_result}")
    
    error = x_est - data['true_etch_rate']
    normal_rmse = np.sqrt(np.mean(error[data['labels'] == 0]**2))
    print(f"EKF RMSE (Normal data): {normal_rmse:.4f}")

if __name__ == '__main__':
    main()
