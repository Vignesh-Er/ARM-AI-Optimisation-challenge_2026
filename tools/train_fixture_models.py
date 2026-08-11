# SPDX-License-Identifier: Apache-2.0
"""Task 2.1: train the Stage-A FIXTURE models (throwaway weights, permanent
shapes — see docs/STATUS.md "Two-stage model plan"). Trained on the current
easy (pre-Phase-4-severity-ladder) dataset at k=1.0, purely to unblock the
exporter/packing/buffer-sizing/bit-exact tests, which depend on dimensions,
not accuracy. Stage-B RELEASE models (same architecture, retrained on the
severity-laddered dataset) come later with no C changes required.

Usage: python tools/train_fixture_models.py
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from phase1_physics.physics_model import PhysicsModel  # noqa: E402
from phase4_tinyml.dataset import generate_cnn_dataset, to_binary_labels  # noqa: E402
from phase4_tinyml.train import train_tier1, train_tier2  # noqa: E402
from tools.verify_bn_fold import assert_bn_folded  # noqa: E402
from tools.write_shape_manifest import main as write_shape_manifest  # noqa: E402


def main():
    physics = PhysicsModel()
    print(f"Generating fixture dataset at window_size={config.WINDOW_SIZE}, k=1.0 ...")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_cnn_dataset(
        physics, n_scenarios=20, n_steps_per=500, seed=config.SEED, k=1.0
    )
    print(f"  X_train={X_train.shape} X_val={X_val.shape} X_test={X_test.shape}")

    print("\n=== Tier-2 (5-class) fixture ===")
    _, tier2_tflite_path, _ = train_tier2(X_train, y_train, X_val, y_val, stage="fixture", epochs=50)
    assert_bn_folded(tier2_tflite_path)
    print(f"BN fold OK: {tier2_tflite_path}")

    print("\n=== Tier-1 (2-class) fixture ===")
    y_train_bin = to_binary_labels(y_train)
    y_val_bin = to_binary_labels(y_val)
    _, tier1_tflite_path, _ = train_tier1(X_train, y_train_bin, X_val, y_val_bin, stage="fixture", epochs=50)
    assert_bn_folded(tier1_tflite_path)
    print(f"BN fold OK: {tier1_tflite_path}")

    print("\n=== Shape manifest ===")
    write_shape_manifest()

    print("\nGATE 2.1 checks: fixture models trained, BN fold verified for both, shapes.json written.")


if __name__ == "__main__":
    main()
