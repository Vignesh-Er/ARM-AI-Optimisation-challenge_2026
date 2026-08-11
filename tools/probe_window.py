# SPDX-License-Identifier: Apache-2.0
"""Task 2.0: sweep window size x fault severity k, train a small float model
per cell, and record balanced accuracy / per-class recall. This settles
PACI_WINDOW_SIZE before any C is written (the ring buffer, every conv output
dimension, and every CMSIS-NN scratch-buffer size depend on it).

Labeling rule: "last" (y = the label of the most recent sample in the
window) is the operative rule everywhere — it's what the deployed cascade
actually sees (paci_step never has future samples), and it's the only rule
under which Phase 4's detection-latency metric stays coherent. "majority"
(the original scheme) is also evaluated here, ONCE, purely as evidence for
why "last" was chosen instead — it is never used for training/evaluation
outside this probe. See phase4_tinyml/dataset.py:extract_windows for both.

Not the Phase 4 severity-ladder rework — this is a throwaway probe. Its
models are never exported; only the numbers in outputs/probe/window_sweep.json
and the window-size decision they produce survive.

Usage: python tools/probe_window.py
"""
import json
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from phase1_physics.physics_model import PhysicsModel  # noqa: E402
from phase1_physics.synthetic_data import generate_full_dataset  # noqa: E402
from phase4_tinyml.dataset import extract_windows  # noqa: E402

_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "probe", "window_sweep.json")

_WINDOWS = [32, 64, 128]
_K_VALUES = [1.0, 0.3, 0.1, 0.03, 0.01]
_N_SCENARIOS = 20
_EPOCHS = 30
_SEED = config.SEED
_DECISION_K = 0.1
_DECISION_MARGIN_PCT = 5.0  # points of balanced accuracy


def _n_steps_per_for_window(window_size):
    """generate_full_dataset's fault_duration is max(20, n_steps // 20),
    fixed relative to n_steps_per, NOT to window_size. Under the "majority"
    labeling rule, whenever fault_duration < window_size / 2, no window can
    ever have fault as its majority label — every window in the scenario is
    labeled Normal by construction, not because the fault is undetectable.
    (This is exactly what happened on the first sweep run, before "last"
    even entered the picture: window=64/128 both came back with a
    degenerate all-Normal test set and a meaningless 1.0 "balanced
    accuracy".) Scale n_steps_per with window_size so fault_duration tracks
    window_size and stays meaningful under "majority" too — kept for the
    evidence comparison even though "last" doesn't have this failure mode
    (a "last" label only needs ONE faulty sample at the window's right
    edge, not half the window).
    """
    return 20 * window_size


def _raw_scenarios(window_size, k, seed):
    """Generate the raw (normalized signal, labels) pair per scenario once;
    both labeling rules are extracted from the same underlying data so the
    "last" vs "majority" comparison isn't confounded by different noise
    draws."""
    physics = PhysicsModel()
    n_steps_per = _n_steps_per_for_window(window_size)
    scenarios = []
    for i in range(_N_SCENARIOS):
        scenario_seed = seed + i
        dataset = generate_full_dataset(physics, n_steps=n_steps_per, seed=scenario_seed, k=k)
        signal = dataset["measured_etch_rate"]
        signal_norm = (signal - config.ETCH_RATE_NOMINAL) / config.NORM_SCALE
        scenarios.append((signal_norm, dataset["labels"]))
    return scenarios


def _build_dataset(scenarios, window_size, rule, seed):
    all_X, all_y = [], []
    for signal_norm, labels in scenarios:
        X_scen, y_scen = extract_windows(signal_norm, labels, window_size=window_size, rule=rule)
        if len(X_scen) == 0:
            continue
        all_X.append(X_scen)
        all_y.append(y_scen)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_train = int(0.7 * len(X))
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _build_model(window_size):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(window_size, 1))
    x = layers.Conv1D(16, kernel_size=5, padding="same", activation=None)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(32, kernel_size=3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation=None)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(config.N_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _evaluate(model, X_test, y_test):
    from sklearn.metrics import balanced_accuracy_score, recall_score

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    per_class_recall = recall_score(
        y_test, y_pred, labels=list(range(config.N_CLASSES)), average=None, zero_division=0
    )
    return balanced_acc, per_class_recall.tolist()


def _train_and_evaluate(scenarios, window_size, rule):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(_SEED)
    X_train, y_train, X_test, y_test = _build_dataset(scenarios, window_size, rule, _SEED)
    model = _build_model(window_size)
    model.fit(X_train, y_train, epochs=_EPOCHS, batch_size=32, verbose=0)
    balanced_acc, per_class_recall = _evaluate(model, X_test, y_test)
    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "balanced_accuracy": float(balanced_acc),
        "per_class_recall": per_class_recall,
    }


def run_sweep():
    results = []
    for window_size in _WINDOWS:
        for k in _K_VALUES:
            scenarios = _raw_scenarios(window_size, k, _SEED)

            last_result = _train_and_evaluate(scenarios, window_size, "last")
            majority_result = _train_and_evaluate(scenarios, window_size, "majority")

            cell = {
                "window": window_size,
                "k": k,
                "last": last_result,
                "majority": majority_result,
            }
            results.append(cell)
            print(f"window={window_size:4d} k={k:<5} "
                  f"last_acc={last_result['balanced_accuracy']:.4f} "
                  f"majority_acc={majority_result['balanced_accuracy']:.4f} "
                  f"last_recall={[round(r, 3) for r in last_result['per_class_recall']]}")

    return results


def apply_decision_rule(results):
    """Smallest W in {32,64,128} such that balanced accuracy at
    k=DECISION_K for 2W exceeds that of W by less than DECISION_MARGIN_PCT
    points (i.e. doubling the window stops paying for itself). If 128 is
    still improving by >= the margin over 64, take 128 anyway and record
    that the sweep doesn't extend further — decided using the "last"
    labeling rule only ("majority" is evidence-only, see module docstring).
    """
    by_window = {}
    for r in results:
        if r["k"] == _DECISION_K:
            by_window[r["window"]] = r["last"]["balanced_accuracy"] * 100.0  # points

    windows = sorted(by_window)
    for w_small, w_big in zip(windows, windows[1:]):
        improvement = by_window[w_big] - by_window[w_small]
        if improvement < _DECISION_MARGIN_PCT:
            return w_small, (
                f"balanced_accuracy(window={w_big}, k={_DECISION_K}, rule=last)="
                f"{by_window[w_big]:.1f}pts improves on window={w_small} "
                f"({by_window[w_small]:.1f}pts) by only {improvement:.1f}pts "
                f"(< {_DECISION_MARGIN_PCT}pt margin) — doubling the window stopped paying for itself"
            )

    largest = windows[-1]
    return largest, (
        f"window={largest} was still improving by >= {_DECISION_MARGIN_PCT}pts over the next "
        f"smaller window at k={_DECISION_K} (rule=last); sweep does not extend past "
        f"{largest} per the frozen decision rule, taking it as-is"
    )


def main():
    results = run_sweep()
    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)
    with open(_OUTPUT_PATH, "w", newline="\n", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {_OUTPUT_PATH}")

    window, reason = apply_decision_rule(results)
    print(f"\nDECISION: PACI_WINDOW_SIZE = {window} ({reason})")


if __name__ == "__main__":
    main()
