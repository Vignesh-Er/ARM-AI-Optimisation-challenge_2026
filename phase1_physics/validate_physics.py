import os
import numpy as np
import matplotlib.pyplot as plt
from phase1_physics.physics_model import PhysicsModel
from phase1_physics.synthetic_data import generate_normal_operation
from config import PLOTS_DIR

def main():
    model = PhysicsModel()
    data = generate_normal_operation(model)
    
    y_true = data['true_etch_rate']
    y_meas = data['measured_etch_rate']
    
    y_pred = np.zeros_like(y_true)
    x = model.nominal_rate
    for i in range(len(y_true)):
        u = np.array([data['params']['pressure'][i], data['params']['temperature'][i], 
                      data['params']['rf_power'][i], data['params']['gas_flow'][i]])
        # Just running the transition function sequentially to track prediction without process noise
        x = model.state_transition(x, u)
        y_pred[i] = x
        
    error = y_pred - y_true
    rmse = np.sqrt(np.mean(error**2))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(y_true, label='Ground Truth')
    ax1.plot(y_pred, label='Physics Prediction', alpha=0.7)
    ax1.set_ylabel('Etch Rate (nm/min)')
    ax1.legend()
    
    ax2.plot(error, label='Prediction Error', color='red')
    ax2.set_ylabel('Error (nm/min)')
    ax2.set_xlabel('Time Step')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'phase1_physics_validation.png'))
    plt.close()
    
    print(f"Physics Model Validation RMSE: {rmse:.4f}")

if __name__ == '__main__':
    main()
