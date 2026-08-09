import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_physics.physics_model import PhysicsModel
from phase1_physics.synthetic_data import generate_full_dataset
from phase2_ekf.ekf import ExtendedKalmanFilter
from phase3_scheduler.scheduler import IntelligentScheduler
import config

def main():
    print("Generating dataset...")
    physics_model = PhysicsModel()
    dataset = generate_full_dataset(physics_model, n_steps=config.N_STEPS, seed=config.SEED)
    
    # Initialize EKF
    x0 = dataset['true_etch_rate'][0]
    P0 = config.P0_VAR
    ekf = ExtendedKalmanFilter(x0, P0, config.Q_VAR, config.R_VAR, physics_model)
    
    scheduler = IntelligentScheduler(
        chi2_threshold=config.CHI2_THRESHOLD_95,
        watchdog_interval=config.WATCHDOG_INTERVAL,
        adaptive_window=config.ADAPTIVE_WINDOW,
        burn_in_steps=config.BURN_IN_STEPS
    )
    
    print("Running EKF and Scheduler over dataset...")
    
    nis_history = []
    
    for k in range(config.N_STEPS):
        u = np.array([
            dataset['params']['pressure'][k],
            dataset['params']['temperature'][k],
            dataset['params']['rf_power'][k],
            dataset['params']['gas_flow'][k]
        ])
        z = dataset['measured_etch_rate'][k]
        
        x_est, P, nis, innovation = ekf.step(u, z)
        nis_history.append(nis)
        
        decision, reason = scheduler.step(nis)
        
    print("Plotting results...")
    
    log = scheduler.log
    decisions = [entry['decision'] for entry in log]
    
    # 1. Timeline showing WAKE/SLEEP
    plt.figure(figsize=(12, 6))
    time = np.arange(config.N_STEPS)
    
    # Highlight faults
    fault_mask = dataset['labels'] > 0
    plt.fill_between(time, 0, 1, where=fault_mask, color='red', alpha=0.3, label='Fault Active', transform=plt.gca().get_xaxis_transform())
    
    wake_mask = np.array(decisions) == 'WAKE_CNN'
    sleep_mask = np.array(decisions) == 'SLEEP'
    
    plt.scatter(time[wake_mask], np.ones(np.sum(wake_mask)) * 0.5, color='orange', label='WAKE_CNN', marker='^', s=10)
    plt.scatter(time[sleep_mask], np.ones(np.sum(sleep_mask)) * 0.5, color='green', label='SLEEP', marker='v', s=10)
    plt.yticks([])
    plt.xlabel('Time Step')
    plt.title('Scheduler Decision Timeline')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'phase3_wake_timeline.png'))
    plt.close()
    
    # 2. NIS over time with threshold
    plt.figure(figsize=(12, 6))
    plt.plot(time, nis_history, label='NIS', color='blue', alpha=0.7)
    plt.axhline(config.CHI2_THRESHOLD_95, color='red', linestyle='--', label='Threshold (95%)')
    plt.yscale('log')
    plt.xlabel('Time Step')
    plt.ylabel('NIS (log scale)')
    plt.title('NIS Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'phase3_nis_decisions.png'))
    plt.close()
    
    # Compute metrics
    total_steps = config.N_STEPS
    normal_mask = dataset['labels'] == 0
    
    wakes = np.sum(wake_mask)
    wake_rate = wakes / total_steps * 100
    
    # False-wake rate: WAKE during normal operation / total normal steps
    false_wakes = np.sum(wake_mask & normal_mask)
    false_wake_rate = false_wakes / np.sum(normal_mask) * 100 if np.sum(normal_mask) > 0 else 0.0
    
    # Missed-anomaly rate: SLEEP during fault / total fault steps
    missed_anomalies = np.sum(sleep_mask & fault_mask)
    missed_anomaly_rate = missed_anomalies / np.sum(fault_mask) * 100 if np.sum(fault_mask) > 0 else 0.0
    
    cnn_reduction = 100.0 - wake_rate
    
    print("\n--- Scheduler Summary ---")
    print(f"Total Steps: {total_steps}")
    print(f"Wake Rate: {wake_rate:.2f}%")
    print(f"CNN Reduction vs Always-On: {cnn_reduction:.2f}%")
    print(f"False-Wake Rate (Normal state but CNN invoked): {false_wake_rate:.2f}%")
    print(f"Missed-Anomaly Rate (Fault state but CNN asleep): {missed_anomaly_rate:.2f}%")
    
if __name__ == '__main__':
    main()
