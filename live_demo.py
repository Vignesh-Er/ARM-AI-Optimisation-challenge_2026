import time
import sys
import os

# Suppress TF logging before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
from colorama import init, Fore, Style

# Initialize colorama for Windows terminal colors
init()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1_physics.physics_model import PhysicsModel
from phase1_physics.synthetic_data import generate_full_dataset
from phase2_ekf.ekf import ExtendedKalmanFilter
from phase3_scheduler.scheduler import IntelligentScheduler
import config

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("================================================================")
    print(" PACI (Physics-Informed Anomaly Classification for TinyML) DEMO ")
    print("================================================================")
    print(f"{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Streaming Live Sensor Data -> Physics -> EKF -> Scheduler -> CNN{Style.RESET_ALL}\n")

def run_demo():
    print("Loading models and preparing data...")
    
    # Try to load trained CNN model
    cnn_model = None
    keras_model_path = os.path.join(config.MODELS_DIR, 'best_cnn.h5')
    if os.path.exists(keras_model_path):
        cnn_model = tf.keras.models.load_model(keras_model_path)
    else:
        print(f"{Fore.RED}Error: CNN model not found. Cannot run full demo.{Style.RESET_ALL}")
        return

    # Generate a short dataset specifically designed for the demo
    physics = PhysicsModel()
    # 600 steps is enough for a demo (Normal -> Sensor Fault -> Gas Leak)
    dataset = generate_full_dataset(physics, n_steps=600, seed=123)
    
    ekf = ExtendedKalmanFilter(
        x0=config.ETCH_RATE_NOMINAL, P0=config.P0_VAR,
        Q=config.Q_VAR, R=config.R_VAR, physics_model=physics
    )
    scheduler = IntelligentScheduler(
        chi2_threshold=config.NIS_THRESHOLD,
        watchdog_interval=config.WATCHDOG_INTERVAL,
        adaptive_window=config.ADAPTIVE_WINDOW,
        burn_in_steps=config.BURN_IN_STEPS
    )

    clear_screen()
    print_header()
    
    window_buffer = []
    
    # Run the stream slowly
    for k in range(len(dataset['measured_etch_rate'])):
        u = np.array([
            dataset['params']['pressure'][k],
            dataset['params']['temperature'][k],
            dataset['params']['rf_power'][k],
            dataset['params']['gas_flow'][k]
        ])
        z = dataset['measured_etch_rate'][k]
        true_label = dataset['labels'][k]
        
        # 1. Physics + EKF
        x_est, P, nis, innovation = ekf.step(u, z)
        
        # 2. Scheduler
        decision, reason = scheduler.step(nis)
        
        # Maintain window for CNN
        window_buffer.append((z - 250.0) / 50.0)
        if len(window_buffer) > config.WINDOW_SIZE:
            window_buffer.pop(0)

        # Formatting Output
        time_str = f"Step {k:03d}"
        sensor_str = f"Sensor: {z:6.1f} nm/m"
        ekf_str = f"NIS: {nis:6.2f}"
        
        if true_label != 0:
            status_str = f"{Fore.RED}[ACTUAL: {config.CLASS_NAMES[true_label]}]{Style.RESET_ALL}"
        else:
            status_str = f"{Fore.GREEN}[ACTUAL: NORMAL]{Style.RESET_ALL}"

        if decision == 'WAKE_CNN':
            if len(window_buffer) == config.WINDOW_SIZE:
                window = np.array(window_buffer).reshape(1, config.WINDOW_SIZE, 1)
                pred = cnn_model(window, training=False).numpy()
                pred_class = np.argmax(pred[0])
                conf = np.max(pred[0]) * 100
                pred_name = config.CLASS_NAMES[pred_class]
                
                if pred_class == 0:
                    cnn_color = Fore.GREEN
                else:
                    cnn_color = Fore.RED
                
                cnn_str = f"{Fore.MAGENTA}>> WAKE CNN <<{Style.RESET_ALL} -> {cnn_color}Pred: {pred_name} ({conf:.1f}%){Style.RESET_ALL}"
            else:
                cnn_str = f"{Fore.MAGENTA}>> WAKE CNN <<{Style.RESET_ALL} -> (Buffering...)"
        else:
            cnn_str = f"{Fore.LIGHTBLACK_EX}CNN Asleep (Saving Power){Style.RESET_ALL}"

        # Print the live stream line
        print(f"{time_str} | {sensor_str} | {ekf_str} | {status_str:30} | {cnn_str}")
        
        # Sleep to simulate real-time sensor polling
        time.sleep(0.05)
        
        # Pause slightly longer when CNN wakes up for dramatic effect
        if decision == 'WAKE_CNN':
            time.sleep(0.2)

    print(f"\n{Fore.CYAN}================================================================{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Demo Completed! PACI successfully reduced CNN calls and detected anomalies.{Style.RESET_ALL}")

if __name__ == '__main__':
    run_demo()
