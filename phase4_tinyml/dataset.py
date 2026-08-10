# SPDX-License-Identifier: Apache-2.0
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1_physics.synthetic_data import generate_full_dataset

def extract_windows(data, labels, window_size=32):
    """Extract sliding windows from time-series.
    Label each window by its majority label.
    
    Returns:
        X: array (n_windows, window_size, 1)
        y: array (n_windows,)
    """
    n_steps = len(data)
    if n_steps < window_size:
        return np.array([]), np.array([])
        
    n_windows = n_steps - window_size + 1
    X = np.zeros((n_windows, window_size, 1))
    y = np.zeros((n_windows,), dtype=int)
    
    for i in range(n_windows):
        X[i, :, 0] = data[i:i+window_size]
        window_labels = labels[i:i+window_size]
        # majority voting
        counts = np.bincount(window_labels)
        y[i] = np.argmax(counts)
        
    return X, y

def generate_cnn_dataset(physics_model, n_scenarios=20, n_steps_per=500, seed=42):
    """Generate a labeled dataset of time-series windows for CNN training.
    
    Generate multiple scenarios with different fault types and locations.
    Extract sliding windows of size WINDOW_SIZE.
    
    5 classes: Normal(0), Sensor Fault(1), Gas Leak(2), Equipment Drift(3), Unexpected(4)
    
    Returns:
        X_train, X_val, X_test: arrays of shape (n_samples, WINDOW_SIZE, 1)
        y_train, y_val, y_test: arrays of class labels
    """
    import config
    np.random.seed(seed)
    
    all_X = []
    all_y = []
    
    for i in range(n_scenarios):
        # Generate a scenario dataset
        scenario_seed = seed + i
        dataset = generate_full_dataset(physics_model, n_steps=n_steps_per, seed=scenario_seed)
        
        # We will use measured_etch_rate as the signal for the CNN, normalized
        signal = dataset['measured_etch_rate']
        
        # Normalization: (measured - ETCH_RATE_NOMINAL) / NORM_SCALE, both
        # from config.py — the paci_core ring-buffer quantizer uses the same
        # two constants (PACI_ETCH_RATE_NOMINAL, PACI_NORM_SCALE) so this
        # normalization can't silently drift out of sync with the C side (D4).
        signal_norm = (signal - config.ETCH_RATE_NOMINAL) / config.NORM_SCALE
        
        X_scen, y_scen = extract_windows(signal_norm, dataset['labels'], window_size=config.WINDOW_SIZE)
        all_X.append(X_scen)
        all_y.append(y_scen)
        
    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)
    
    # Shuffle
    indices = np.arange(len(X_full))
    np.random.shuffle(indices)
    X_full = X_full[indices]
    y_full = y_full[indices]
    
    # Split: 60/20/20
    n = len(X_full)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    
    X_train = X_full[:n_train]
    y_train = y_full[:n_train]
    
    X_val = X_full[n_train:n_train+n_val]
    y_val = y_full[n_train:n_train+n_val]
    
    X_test = X_full[n_train+n_val:]
    y_test = y_full[n_train+n_val:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
