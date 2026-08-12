# SPDX-License-Identifier: Apache-2.0
import numpy as np
from config import *
from phase1_physics.physics_model import PhysicsModel

def generate_process_params(n_steps, seed=42):
    """Generate slowly varying, realistic process parameters.
    
    Use smoothed random walks (low-pass filtered) to simulate
    realistic process parameter variations within the ranges
    defined in config.py.
    
    Returns:
        dict with keys 'pressure', 'temperature', 'rf_power', 'gas_flow',
        each a numpy array of shape (n_steps,)
    """
    np.random.seed(seed)
    
    def smooth_walk(start, min_val, max_val, length):
        walk = np.zeros(length)
        walk[0] = start
        target = np.random.uniform(min_val, max_val)
        for i in range(1, length):
            if i % 100 == 0:
                target = np.random.uniform(min_val, max_val)
            walk[i] = walk[i-1] + 0.05 * (target - walk[i-1]) + np.random.normal(0, (max_val - min_val) * 0.01)
            walk[i] = np.clip(walk[i], min_val, max_val)
        return walk
        
    p_walk = smooth_walk(P_REF, PRESSURE_RANGE[0], PRESSURE_RANGE[1], n_steps)
    t_walk = smooth_walk(T_REF, TEMPERATURE_RANGE[0], TEMPERATURE_RANGE[1], n_steps)
    w_walk = smooth_walk(W_REF, RF_POWER_RANGE[0], RF_POWER_RANGE[1], n_steps)
    f_walk = smooth_walk(F_REF, GAS_FLOW_RANGE[0], GAS_FLOW_RANGE[1], n_steps)
    
    return {
        'pressure': p_walk,
        'temperature': t_walk,
        'rf_power': w_walk,
        'gas_flow': f_walk
    }

def generate_normal_operation(physics_model, n_steps=N_STEPS, sensor_noise_std=SENSOR_NOISE_STD, seed=SEED):
    """Generate normal (no fault) operation data.
    
    - Compute true etch rate using physics model with process noise
    - Add measurement noise to get sensor readings
    
    Returns:
        dict with:
          'params': process params dict
          'true_etch_rate': array (n_steps,)
          'measured_etch_rate': array (n_steps,)
          'labels': array of zeros (n_steps,) — all normal
    """
    np.random.seed(seed)
    params = generate_process_params(n_steps, seed)
    
    true_etch_rate = np.zeros(n_steps)
    measured_etch_rate = np.zeros(n_steps)
    labels = np.zeros(n_steps, dtype=int)
    
    x = physics_model.nominal_rate
    for i in range(n_steps):
        u = np.array([params['pressure'][i], params['temperature'][i], 
                      params['rf_power'][i], params['gas_flow'][i]])
        
        rate_pred = physics_model.etch_rate(u) + np.random.normal(0, PROCESS_NOISE_STD)
        x = (1 - physics_model.tau) * x + physics_model.tau * rate_pred
        
        true_etch_rate[i] = x
        measured_etch_rate[i] = x + np.random.normal(0, sensor_noise_std)
        
    return {
        'params': params,
        'true_etch_rate': true_etch_rate,
        'measured_etch_rate': measured_etch_rate,
        'labels': labels
    }

def inject_fault(data, fault_type, start_step, duration, seed=SEED, k=1.0):
    """Inject a specific fault into the data, scaled by severity k.

    fault_type options:
    - 'sensor_fault': stuck-at-last-reading. k controls how much of the
      residual (pre-fault) signal survives: k=1.0 is a hard stick (the
      reading freezes at the value one step before the fault and stays
      there with zero noise); k=0 leaves the reading untouched (no fault).
      NOTE: this replaces the old flatline-to-0.0 definition, which was a
      ~125-sigma jump and physically implausible for a stuck sensor. This
      is NOT byte-for-byte equivalent to the old behaviour at any k,
      including k=1.0 — see tests/test_fault_injection_k1_matches_baseline.py,
      which documents this as the one deliberate exception to the
      "k=1.0 reproduces today's behaviour" rule that the other three fault
      types are held to.
    - 'gas_leak': sudden drop in gas flow by k * GAS_LEAK_FLOW_DROP,
      affects true etch rate. k=1.0 matches the original fixed 40% drop
      byte-for-byte.
    - 'equipment_drift': slow linear drift in etch rate
      (+k * DRIFT_RATE per step). k=1.0 matches the original fixed 5%/step
      byte-for-byte.
    - 'unexpected_deviation': random large spikes
      (k * UNEXPECTED_DEVIATION_MAGNITUDE times noise std). k=1.0 matches
      the original fixed 3x magnitude byte-for-byte.

    Modifies data in-place and updates labels.
    Labels: 0=Normal, 1=Sensor Fault, 2=Gas Leak, 3=Equipment Drift, 4=Unexpected Deviation

    Returns:
        modified data dict with updated labels
    """
    np.random.seed(seed)
    end_step = min(start_step + duration, len(data['labels']))

    if fault_type == 'sensor_fault':
        stuck_value = (data['measured_etch_rate'][start_step - 1]
                       if start_step > 0 else data['measured_etch_rate'][start_step])
        for i in range(start_step, end_step):
            residual_reading = data['measured_etch_rate'][i]
            data['measured_etch_rate'][i] = (1.0 - k) * residual_reading + k * stuck_value
        data['labels'][start_step:end_step] = 1

    elif fault_type == 'gas_leak':
        # Drop gas flow
        drop_factor = (1.0 - GAS_LEAK_FLOW_DROP) if k == 1.0 else (1.0 - k * GAS_LEAK_FLOW_DROP)
        data['params']['gas_flow'][start_step:end_step] *= drop_factor
        
        # Recalculate true and measured
        model = PhysicsModel()
        x = data['true_etch_rate'][start_step-1] if start_step > 0 else model.nominal_rate
        for i in range(start_step, end_step):
            u = np.array([data['params']['pressure'][i], data['params']['temperature'][i],
                          data['params']['rf_power'][i], data['params']['gas_flow'][i]])
            rate_pred = model.etch_rate(u) + np.random.normal(0, PROCESS_NOISE_STD)
            x = (1 - model.tau) * x + model.tau * rate_pred
            data['true_etch_rate'][i] = x
            data['measured_etch_rate'][i] = x + np.random.normal(0, SENSOR_NOISE_STD)

        # recover after the leak
        for i in range(end_step, len(data['labels'])):
            if data['labels'][i] == 0:
                u = np.array([data['params']['pressure'][i], data['params']['temperature'][i],
                              data['params']['rf_power'][i], data['params']['gas_flow'][i]])
                rate_pred = model.etch_rate(u) + np.random.normal(0, PROCESS_NOISE_STD)
                x = (1 - model.tau) * x + model.tau * rate_pred
                data['true_etch_rate'][i] = x
                data['measured_etch_rate'][i] = x + np.random.normal(0, SENSOR_NOISE_STD)
            else:
                break

        data['labels'][start_step:end_step] = 2

    elif fault_type == 'equipment_drift':
        for i in range(start_step, end_step):
            drift_factor = (1.0 + DRIFT_RATE * (i - start_step)) if k == 1.0 else (1.0 + k * DRIFT_RATE * (i - start_step))
            data['true_etch_rate'][i] *= drift_factor
            data['measured_etch_rate'][i] = data['true_etch_rate'][i] + np.random.normal(0, SENSOR_NOISE_STD)
        data['labels'][start_step:end_step] = 3

    elif fault_type == 'unexpected_deviation':
        for i in range(start_step, end_step):
            noise_std = (SENSOR_NOISE_STD * UNEXPECTED_DEVIATION_MAGNITUDE) if k == 1.0 else (SENSOR_NOISE_STD * k * UNEXPECTED_DEVIATION_MAGNITUDE)
            data['measured_etch_rate'][i] += np.random.normal(0, noise_std)
        data['labels'][start_step:end_step] = 4

    return data

def generate_full_dataset(physics_model, n_steps=N_STEPS, seed=SEED, k=1.0):
    """Generate a complete dataset with normal operation + all fault types.

    Layout: divide the time series into segments, insert each fault type
    with normal gaps between them. Fault positions scale with n_steps.

    k: severity multiplier passed through to every inject_fault() call
    (see its docstring). k=1.0 reproduces the original fixed-magnitude
    faults for gas_leak/equipment_drift/unexpected_deviation; sensor_fault
    is a deliberate redefinition (stuck-at-last-reading) at every k.

    Returns:
        dict with all data + labels
    """
    data = generate_normal_operation(physics_model, n_steps, SENSOR_NOISE_STD, seed)

    # Scale fault locations proportionally to n_steps
    # Each fault gets ~1/5 of the timeline, with the first 20% being normal
    fault_duration = max(20, n_steps // 20)  # ~5% of total steps per fault
    gap = max(10, (n_steps - 4 * fault_duration) // 5)  # equal gaps between faults

    faults = [
        ('sensor_fault',          gap,                              fault_duration),
        ('gas_leak',              gap + 1 * (fault_duration + gap), fault_duration),
        ('equipment_drift',       gap + 2 * (fault_duration + gap), fault_duration),
        ('unexpected_deviation',  gap + 3 * (fault_duration + gap), fault_duration),
    ]

    for f_type, start, dur in faults:
        if start + dur <= n_steps:
            data = inject_fault(data, f_type, start, dur, seed, k=k)

    return data
