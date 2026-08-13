# SPDX-License-Identifier: Apache-2.0
"""Train the Stage-B RELEASE models (same architecture, retrained on the
severity-laddered dataset — see docs/STATUS.md "Two-stage model plan").

Usage: python tools/train_release_models.py
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


def main():
    physics = PhysicsModel()
    print(f"Generating release dataset at window_size={config.WINDOW_SIZE}, k=1.0 ...")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_cnn_dataset(
        physics, n_scenarios=40, n_steps_per=1000, seed=config.SEED, k=1.0
    )
    print(f"  X_train={X_train.shape} X_val={X_val.shape} X_test={X_test.shape}")

    print("\n=== Tier-2 (5-class) release ===")
    _, tier2_tflite_path, _ = train_tier2(X_train, y_train, X_val, y_val, stage="release", epochs=50)
    assert_bn_folded(tier2_tflite_path)
    print(f"BN fold OK: {tier2_tflite_path}")

    print("\n=== Tier-1 (2-class) release ===")
    y_train_bin = to_binary_labels(y_train)
    y_val_bin = to_binary_labels(y_val)
    _, tier1_tflite_path, _ = train_tier1(X_train, y_train_bin, X_val, y_val_bin, stage="release", epochs=50)
    assert_bn_folded(tier1_tflite_path)
    print(f"BN fold OK: {tier1_tflite_path}")

    print("\nRelease models trained and BN fold verified for both.")


if __name__ == "__main__":
    main()
