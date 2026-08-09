"""
Phase 6 — Benchmark Runner

Runs all baseline gating methods + PACI on the same test dataset
and collects comprehensive comparison metrics.
"""
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_physics.physics_model import PhysicsModel
from phase1_physics.synthetic_data import generate_full_dataset
from phase2_ekf.ekf import ExtendedKalmanFilter
from phase3_scheduler.scheduler import IntelligentScheduler
from phase6_benchmark.baselines import (
    AlwaysOnBaseline, VarianceThresholdGate, MovingAverageGate,
    CUSUMGate, KalmanNoPhysicsGate
)
import config


def run_paci_gating(dataset):
    """Run PACI pipeline (physics + EKF + NIS gating) and return metrics."""
    physics = PhysicsModel()
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
    
    decisions = []
    n_steps = len(dataset['measured_etch_rate'])
    
    start_time = time.perf_counter()
    for k in range(n_steps):
        u = np.array([
            dataset['params']['pressure'][k],
            dataset['params']['temperature'][k],
            dataset['params']['rf_power'][k],
            dataset['params']['gas_flow'][k]
        ])
        z = dataset['measured_etch_rate'][k]
        
        x_est, P, nis, innovation = ekf.step(u, z)
        decision, reason = scheduler.step(nis)
        decisions.append(decision)
    elapsed = time.perf_counter() - start_time
    
    return decisions, elapsed


def run_baseline(baseline_class, dataset):
    """Run a baseline gating method and return decisions."""
    gate = baseline_class()
    decisions = []
    n_steps = len(dataset['measured_etch_rate'])
    
    start_time = time.perf_counter()
    for k in range(n_steps):
        z = dataset['measured_etch_rate'][k]
        decision, reason = gate.step(z)
        decisions.append(decision)
    elapsed = time.perf_counter() - start_time
    
    return decisions, elapsed


def compute_metrics(decisions, labels):
    """Compute gating metrics from decisions and ground truth labels."""
    n = len(decisions)
    wake_mask = np.array([d == 'WAKE_CNN' for d in decisions])
    fault_mask = labels > 0
    normal_mask = labels == 0
    
    n_wakes = np.sum(wake_mask)
    wake_rate = n_wakes / n
    cnn_reduction = 1.0 - wake_rate
    
    # False-wake rate: WAKE during normal
    false_wakes = np.sum(wake_mask & normal_mask)
    false_wake_rate = false_wakes / np.sum(normal_mask) if np.sum(normal_mask) > 0 else 0
    
    # Missed-anomaly rate: SLEEP during fault (per-step)
    missed = np.sum(~wake_mask & fault_mask)
    missed_rate = missed / np.sum(fault_mask) if np.sum(fault_mask) > 0 else 0
    
    # Fault detection rate: for each contiguous fault region,
    # did at least one WAKE occur?
    fault_types = [1, 2, 3, 4]
    detected_faults = 0
    total_faults = 0
    for ft in fault_types:
        ft_mask = labels == ft
        if np.any(ft_mask):
            total_faults += 1
            if np.any(wake_mask & ft_mask):
                detected_faults += 1
    
    fault_detection_rate = detected_faults / total_faults if total_faults > 0 else 1.0
    
    # Simulated energy (relative units)
    # Physics+EKF cost: 1 unit per step, CNN cost: 50 units per invocation
    physics_cost_per_step = 1.0
    cnn_cost_per_invocation = 50.0
    total_energy = n * physics_cost_per_step + n_wakes * cnn_cost_per_invocation
    always_on_energy = n * physics_cost_per_step + n * cnn_cost_per_invocation
    energy_saving = 1.0 - (total_energy / always_on_energy) if always_on_energy > 0 else 0
    
    return {
        'cnn_invocations': int(n_wakes),
        'wake_rate': wake_rate,
        'cnn_reduction_pct': cnn_reduction * 100,
        'false_wake_rate': false_wake_rate,
        'missed_anomaly_rate_step': missed_rate,
        'fault_detection_rate': fault_detection_rate,
        'simulated_energy': total_energy,
        'energy_saving_pct': energy_saving * 100,
    }


def main():
    """Run all benchmarks and print comparison table."""
    print("=" * 80)
    print("PACI Phase 6 — Comprehensive Benchmarking")
    print("=" * 80)
    
    # Generate test dataset
    physics = PhysicsModel()
    dataset = generate_full_dataset(physics, n_steps=config.N_STEPS, seed=99)
    labels = dataset['labels']
    
    print(f"Dataset: {config.N_STEPS} steps, {np.sum(labels > 0)} fault steps, "
          f"{np.sum(labels == 0)} normal steps")
    print(f"Fault types present: {np.unique(labels[labels > 0])}")
    print()
    
    # Run all methods
    results = {}
    
    # PACI
    print("Running PACI (Physics + EKF + NIS Gate)...", end=" ")
    decisions, elapsed = run_paci_gating(dataset)
    metrics = compute_metrics(decisions, labels)
    metrics['latency_ms'] = elapsed * 1000 / config.N_STEPS
    results['PACI'] = metrics
    print(f"Done ({elapsed*1000:.1f}ms total)")
    
    # Baselines
    baseline_classes = {
        'Always-On CNN': AlwaysOnBaseline,
        'Variance Threshold': VarianceThresholdGate,
        'Moving Average': MovingAverageGate,
        'CUSUM Detector': CUSUMGate,
        'Kalman (No Physics)': KalmanNoPhysicsGate,
    }
    
    for name, cls in baseline_classes.items():
        print(f"Running {name}...", end=" ")
        decisions, elapsed = run_baseline(cls, dataset)
        metrics = compute_metrics(decisions, labels)
        metrics['latency_ms'] = elapsed * 1000 / config.N_STEPS
        results[name] = metrics
        print(f"Done ({elapsed*1000:.1f}ms total)")
    
    # Print comparison table
    print("\n" + "=" * 110)
    print(f"{'Method':<25} {'CNN Runs':>10} {'Reduction':>10} {'Wake Rate':>10} "
          f"{'Fault Det.':>10} {'False Wake':>10} {'Energy Save':>12}")
    print("-" * 110)
    
    for name, m in results.items():
        print(f"{name:<25} {m['cnn_invocations']:>10} {m['cnn_reduction_pct']:>9.1f}% "
              f"{m['wake_rate']:>9.1%} {m['fault_detection_rate']:>9.1%} "
              f"{m['false_wake_rate']:>9.1%} {m['energy_saving_pct']:>10.1f}%")
    
    print("=" * 110)
    
    # Highlight PACI results
    paci = results['PACI']
    print(f"\n>>> PACI Headline: Reduced CNN execution by {paci['cnn_reduction_pct']:.0f}% "
          f"with {paci['fault_detection_rate']:.0%} fault detection rate")
    
    # Save results to file
    import json
    report_path = os.path.join(config.REPORTS_DIR, 'benchmark_results.json')
    # Convert numpy types for JSON serialization
    clean_results = {}
    for method, metrics in results.items():
        clean_results[method] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                  for k, v in metrics.items()}
    
    with open(report_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nResults saved to: {report_path}")
    
    return results


if __name__ == '__main__':
    main()
