"""
PACI End-to-End Cascade Pipeline
sensor → physics → EKF → scheduler → CNN

Runs the full inference cascade and collects comprehensive metrics.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_physics.physics_model import PhysicsModel
from phase1_physics.synthetic_data import generate_full_dataset
from phase2_ekf.ekf import ExtendedKalmanFilter
from phase3_scheduler.scheduler import IntelligentScheduler
import config


class CascadePipeline:
    """Full PACI cascade: sensor → physics → EKF → scheduler → CNN."""
    
    def __init__(self, cnn_model=None, use_gating=True):
        """
        Args:
            cnn_model: Trained Keras model for fault classification (or None for scheduler-only eval)
            use_gating: If False, CNN runs every step (always-on baseline)
        """
        self.physics = PhysicsModel()
        self.ekf = ExtendedKalmanFilter(
            x0=config.ETCH_RATE_NOMINAL,
            P0=config.P0_VAR,
            Q=config.Q_VAR,
            R=config.R_VAR,
            physics_model=self.physics
        )
        self.scheduler = IntelligentScheduler(
            chi2_threshold=config.NIS_THRESHOLD,
            watchdog_interval=config.WATCHDOG_INTERVAL,
            adaptive_window=config.ADAPTIVE_WINDOW,
            burn_in_steps=config.BURN_IN_STEPS
        )
        self.cnn_model = cnn_model
        self.use_gating = use_gating
        
        # Metrics
        self.results = {
            'state_estimates': [],
            'nis_values': [],
            'innovations': [],
            'decisions': [],
            'reasons': [],
            'cnn_predictions': [],
            'cnn_invocations': 0,
            'total_steps': 0,
        }
    
    def run(self, dataset):
        """Run the cascade over a full dataset.
        
        Args:
            dataset: dict from generate_full_dataset()
        
        Returns:
            dict of results and metrics
        """
        n_steps = len(dataset['measured_etch_rate'])
        self.results['total_steps'] = n_steps
        
        # Window buffer for CNN
        window_buffer = []
        
        for k in range(n_steps):
            u = np.array([
                dataset['params']['pressure'][k],
                dataset['params']['temperature'][k],
                dataset['params']['rf_power'][k],
                dataset['params']['gas_flow'][k]
            ])
            z = dataset['measured_etch_rate'][k]
            
            # Step 1: Physics + EKF
            x_est, P, nis, innovation = self.ekf.step(u, z)
            
            self.results['state_estimates'].append(x_est)
            self.results['nis_values'].append(nis)
            self.results['innovations'].append(innovation)
            
            # Step 2: Scheduler decision
            if self.use_gating:
                decision, reason = self.scheduler.step(nis)
            else:
                decision, reason = 'WAKE_CNN', 'always_on'
            
            self.results['decisions'].append(decision)
            self.results['reasons'].append(reason)
            
            # Step 3: CNN inference (if woken)
            if decision == 'WAKE_CNN' and self.cnn_model is not None:
                self.results['cnn_invocations'] += 1
                
                # Build window from recent measurements
                window_buffer.append((z - 250.0) / 50.0)  # normalise
                if len(window_buffer) > config.WINDOW_SIZE:
                    window_buffer = window_buffer[-config.WINDOW_SIZE:]
                
                if len(window_buffer) >= config.WINDOW_SIZE:
                    window = np.array(window_buffer[-config.WINDOW_SIZE:]).reshape(1, config.WINDOW_SIZE, 1)
                    # Use direct __call__ for fast single-sample inference instead of .predict() overhead
                    pred = self.cnn_model(window, training=False).numpy()
                    pred_class = np.argmax(pred[0])
                    self.results['cnn_predictions'].append({
                        'step': k,
                        'prediction': int(pred_class),
                        'confidence': float(np.max(pred[0])),
                        'true_label': int(dataset['labels'][k])
                    })
                else:
                    # Not enough data for CNN yet
                    self.results['cnn_predictions'].append({
                        'step': k,
                        'prediction': 0,
                        'confidence': 0.0,
                        'true_label': int(dataset['labels'][k])
                    })
            else:
                # CNN sleeping — just maintain window buffer
                window_buffer.append((z - 250.0) / 50.0)
                if len(window_buffer) > config.WINDOW_SIZE:
                    window_buffer = window_buffer[-config.WINDOW_SIZE:]
        
        # Compute summary metrics
        self._compute_metrics(dataset)
        return self.results
    
    def _compute_metrics(self, dataset):
        """Compute summary performance metrics."""
        labels = dataset['labels']
        n = len(labels)
        
        cnn_count = self.results['cnn_invocations']
        self.results['cnn_reduction_pct'] = (1.0 - cnn_count / n) * 100 if n > 0 else 0
        self.results['wake_rate_pct'] = (cnn_count / n) * 100 if n > 0 else 0
        
        # Detection metrics (from CNN predictions)
        if self.results['cnn_predictions']:
            preds = self.results['cnn_predictions']
            true_labels = [p['true_label'] for p in preds]
            pred_labels = [p['prediction'] for p in preds]
            
            # Per-step accuracy of CNN when invoked
            correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
            self.results['cnn_accuracy'] = correct / len(preds) if preds else 0
            
            # Anomaly detection: any non-zero prediction counts as detected
            # Check which fault windows had at least one correct detection
            fault_steps = set(np.where(labels > 0)[0])
            detected_steps = set(p['step'] for p in preds if p['prediction'] > 0)
            
            if fault_steps:
                detected_fault_steps = fault_steps & detected_steps
                self.results['anomaly_recall'] = len(detected_fault_steps) / len(fault_steps)
            else:
                self.results['anomaly_recall'] = 1.0
        
        self.results['state_estimate_rmse'] = np.sqrt(
            np.mean((np.array(self.results['state_estimates']) - dataset['true_etch_rate']) ** 2)
        )


def main():
    """Run the full cascade pipeline and report metrics."""
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("PACI End-to-End Cascade Pipeline")
    print("=" * 60)
    
    # Try to load trained CNN model
    cnn_model = None
    keras_model_path = os.path.join(config.MODELS_DIR, 'best_cnn.h5')
    if os.path.exists(keras_model_path):
        import tensorflow as tf
        cnn_model = tf.keras.models.load_model(keras_model_path)
        print(f"Loaded CNN model from {keras_model_path}")
    else:
        print("WARNING: No trained CNN model found. Running scheduler-only evaluation.")
    
    # Generate test dataset
    physics = PhysicsModel()
    dataset = generate_full_dataset(physics, n_steps=config.N_STEPS, seed=99)  # Different seed from training
    
    # Run WITH gating (PACI)
    print("\n--- Running PACI (with gating) ---")
    pipeline_gated = CascadePipeline(cnn_model=cnn_model, use_gating=True)
    results_gated = pipeline_gated.run(dataset)
    
    # Run WITHOUT gating (always-on baseline)
    print("\n--- Running Always-On CNN (no gating) ---")
    pipeline_always = CascadePipeline(cnn_model=cnn_model, use_gating=False)
    results_always = pipeline_always.run(dataset)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<35} {'PACI':>12} {'Always-On':>12}")
    print("-" * 60)
    print(f"{'CNN Invocations':<35} {results_gated['cnn_invocations']:>12} {results_always['cnn_invocations']:>12}")
    print(f"{'CNN Reduction (%)':<35} {results_gated['cnn_reduction_pct']:>11.1f}% {'0.0%':>12}")
    print(f"{'State Estimate RMSE':<35} {results_gated['state_estimate_rmse']:>12.2f} {results_always['state_estimate_rmse']:>12.2f}")
    
    if cnn_model is not None:
        cnn_acc_g = results_gated.get('cnn_accuracy', 0)
        cnn_acc_a = results_always.get('cnn_accuracy', 0)
        recall_g = results_gated.get('anomaly_recall', 0)
        recall_a = results_always.get('anomaly_recall', 0)
        print(f"{'CNN Accuracy (when invoked)':<35} {cnn_acc_g:>11.1%} {cnn_acc_a:>11.1%}")
        print(f"{'Anomaly Recall':<35} {recall_g:>11.1%} {recall_a:>11.1%}")
    
    print("=" * 60)
    
    # Generate pipeline visualization
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    steps = np.arange(config.N_STEPS)
    
    # Panel 1: State estimate vs truth
    axes[0].plot(steps, dataset['true_etch_rate'], 'b-', alpha=0.5, label='True')
    axes[0].plot(steps, results_gated['state_estimates'], 'r-', alpha=0.7, label='EKF Estimate')
    axes[0].set_ylabel('Etch Rate (nm/min)')
    axes[0].legend()
    axes[0].set_title('PACI Pipeline — End-to-End Results')
    
    # Panel 2: NIS with threshold
    axes[1].semilogy(steps, results_gated['nis_values'], 'b-', alpha=0.5, linewidth=0.5)
    axes[1].axhline(y=config.NIS_THRESHOLD, color='r', linestyle='--', label=f'χ² threshold ({config.NIS_THRESHOLD})')
    axes[1].set_ylabel('NIS (log)')
    axes[1].legend()
    
    # Panel 3: CNN decisions
    decisions_binary = [1 if d == 'WAKE_CNN' else 0 for d in results_gated['decisions']]
    axes[2].fill_between(steps, decisions_binary, alpha=0.3, color='orange', label='CNN Active')
    # Highlight faults
    fault_mask = dataset['labels'] > 0
    axes[2].fill_between(steps, fault_mask.astype(int) * 0.5, alpha=0.3, color='red', label='Fault Region')
    axes[2].set_ylabel('Activity')
    axes[2].legend()
    axes[2].set_ylim(-0.1, 1.3)
    
    # Panel 4: Labels
    axes[3].plot(steps, dataset['labels'], 'k-', linewidth=0.5)
    axes[3].set_ylabel('Fault Label')
    axes[3].set_xlabel('Time Step')
    axes[3].set_yticks(range(5))
    axes[3].set_yticklabels(config.CLASS_NAMES, fontsize=8)
    
    plt.tight_layout()
    plot_path = os.path.join(config.PLOTS_DIR, 'pipeline_end_to_end.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPipeline plot saved to: {plot_path}")


if __name__ == '__main__':
    main()
