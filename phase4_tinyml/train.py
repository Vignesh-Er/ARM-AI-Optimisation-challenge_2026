# SPDX-License-Identifier: Apache-2.0
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase4_tinyml.model import create_tier1_model, create_tier2_model


def _representative_dataset(X_train, n_samples=100):
    def gen():
        for i in range(min(n_samples, len(X_train))):
            yield [X_train[i:i + 1].astype(np.float32)]
    return gen


def _train_and_quantize(model, X_train, y_train, X_val, y_val, model_name, stage, epochs, batch_size, from_logits):
    """Train a Keras model and convert it to a full-integer INT8 TFLite
    model. `stage` must be 'fixture' or 'release' (Phase 2, Task 2.0's
    two-stage plan) — it's load-bearing in the output filename so a
    throwaway fixture model can never be mistaken for a real result.

    `from_logits` must be True for Tier-1 (its graph ends at a raw Dense(2)
    output, no softmax) and False for Tier-2 (its graph ends in Softmax()).
    """
    if stage not in ("fixture", "release"):
        raise ValueError(f"stage must be 'fixture' or 'release', got {stage!r}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.CNN_LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits),
        metrics=["accuracy"],
    )

    keras_model_path = os.path.join(config.MODELS_DIR, f"{model_name}_{stage}.h5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        keras_model_path, monitor="val_accuracy", save_best_only=True, verbose=0
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print(f"Training {model_name} ({stage})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=0,
    )

    print(f"Quantizing {model_name} ({stage}) to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_model_path = os.path.join(config.MODELS_DIR, f"{model_name}_{stage}.tflite")
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    if os.path.exists(keras_model_path):
        print(f"  Keras model size: {os.path.getsize(keras_model_path) / 1024:.2f} KB")
    print(f"  TFLite INT8 model size: {os.path.getsize(tflite_model_path) / 1024:.2f} KB")

    return model, tflite_model_path, history


def train_tier2(X_train, y_train, X_val, y_val, stage="fixture", epochs=50, batch_size=32):
    """5-class classifier. y_train/y_val use the full 0..4 label space."""
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_tier2_model(input_shape=input_shape, n_classes=config.N_CLASSES)
    return _train_and_quantize(
        model, X_train, y_train, X_val, y_val, "tier2", stage, epochs, batch_size, from_logits=False
    )


def train_tier1(X_train, y_train_binary, X_val, y_val_binary, stage="fixture", epochs=50, batch_size=32):
    """2-class (normal vs anomalous) screen. y_train_binary/y_val_binary must
    already be collapsed to {0, 1} — see phase4_tinyml.dataset.to_binary_labels.
    Quantized to INT8 here; Task 2.2/2.3's exporter re-quantizes the
    designated layers to INT4 on top of this INT8 baseline.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_tier1_model(input_shape=input_shape)
    return _train_and_quantize(
        model, X_train, y_train_binary, X_val, y_val_binary, "tier1", stage, epochs, batch_size, from_logits=True
    )
