# SPDX-License-Identifier: Apache-2.0
"""
PACI Configuration — All hyperparameters and physical constants.
Centralised here so every module draws from one source of truth.
"""
import numpy as np

# ─────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────
KB_EV = 8.617e-5          # Boltzmann constant (eV/K)

# ─────────────────────────────────────────────
# Deal-Grove-derived plasma etch model params
# ─────────────────────────────────────────────
PHYSICS_K0 = 2.5          # rate constant (nm/min at reference conditions)
PHYSICS_ALPHA = 0.6       # pressure exponent
PHYSICS_BETA = 0.4        # gas flow exponent
PHYSICS_GAMMA = 0.3       # RF power exponent
PHYSICS_EA = 0.15         # activation energy (eV)

# Reference operating point (normalisation)
P_REF = 50.0              # pressure (mTorr)
T_REF = 323.15            # temperature (K) — 50 °C
W_REF = 500.0             # RF power (W)
F_REF = 100.0             # gas flow (sccm)

# Etch-rate range for realistic scaling (nm/min)
ETCH_RATE_NOMINAL = 250.0     # nominal etch rate at reference point

# State-transition dynamics
TAU = 0.3                 # response factor  (0 < tau < 1)
DT = 1.0                  # timestep (seconds)

# ─────────────────────────────────────────────
# EKF parameters
# ─────────────────────────────────────────────
Q_VAR = 0.5               # process noise variance
R_VAR = 4.0               # measurement noise variance
P0_VAR = 10.0             # initial state covariance

# ─────────────────────────────────────────────
# Scheduler / NIS gating
# ─────────────────────────────────────────────
CHI2_THRESHOLD_95 = 3.841     # χ²(1, 0.95)  — 1 DOF, 95 % confidence
CHI2_THRESHOLD_99 = 6.635     # χ²(1, 0.99)  — 1 DOF, 99 % confidence
NIS_THRESHOLD = CHI2_THRESHOLD_95   # default gate threshold
WATCHDOG_INTERVAL = 50        # max cycles between forced CNN runs
ADAPTIVE_WINDOW = 500         # rolling window for adaptive Q/R
BURN_IN_STEPS = 30            # EKF warm-up before gating activates

# ─────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────
N_STEPS = 2000                # total simulation timesteps
SENSOR_NOISE_STD = 2.0        # measurement noise σ  (nm/min)
PROCESS_NOISE_STD = np.sqrt(Q_VAR)
SEED = 42                     # reproducibility

# Nominal process-parameter ranges (uniform random around operating point)
PRESSURE_RANGE = (30.0, 70.0)       # mTorr
TEMPERATURE_RANGE = (303.15, 343.15) # K  (30–70 °C)
RF_POWER_RANGE = (300.0, 700.0)     # W
GAS_FLOW_RANGE = (60.0, 140.0)     # sccm

# ─────────────────────────────────────────────
# CNN / TinyML
# ─────────────────────────────────────────────
WINDOW_SIZE = 64              # 1-D CNN input window length — frozen by
                                # tools/probe_window.py (Task 2.0); see
                                # outputs/probe/window_sweep.json and
                                # docs/STATUS.md for the sweep + decision.
N_CLASSES = 5
NORM_SCALE = 50.0              # CNN input normalization: (measured - ETCH_RATE_NOMINAL) / NORM_SCALE
                                # single source of truth for both phase4_tinyml/dataset.py
                                # and paci_core's ring-buffer quantizer (D4)
CLASS_NAMES = [
    "Normal",
    "Sensor Fault",
    "Gas Leak",
    "Equipment Drift",
    "Unexpected Deviation",
]
CNN_EPOCHS = 50
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 1e-3

# Fault-injection parameters
FAULT_DURATION_RANGE = (40, 120)     # steps
SENSOR_FAULT_FLATLINE_VALUE = 0.0    # sensor reads zero
GAS_LEAK_FLOW_DROP = 0.4            # fraction drop in gas flow
DRIFT_RATE = 0.05                    # fractional drift per step
UNEXPECTED_DEVIATION_MAGNITUDE = 3.0 # multiplier on noise std

# ─────────────────────────────────────────────
# Output paths
# ─────────────────────────────────────────────
import os as _os
_PROJECT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
OUTPUT_DIR = _os.path.join(_PROJECT_ROOT, "outputs")
PLOTS_DIR = _os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = _os.path.join(OUTPUT_DIR, "models")
REPORTS_DIR = _os.path.join(OUTPUT_DIR, "reports")

# Create output dirs on import
for _d in (OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR):
    _os.makedirs(_d, exist_ok=True)
