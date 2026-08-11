# SPDX-License-Identifier: Apache-2.0
"""Task 2.2 (timeboxed ~45 min): does tensorflow==2.21.0's TFLiteConverter
actually emit TensorType_INT4 tensors for a full-integer quantized model, or
does it silently fall back to int8? (tensorflow/tensorflow#64193 tracked this
as an open gap in the converter's int4-emission path.) This is independent
of which window size Task 2.0 settles on — it's a converter capability
probe, not an accuracy probe — so it uses a fixed small window and is
disposable; nothing here is exported or trained for real.

Usage: python tools/probe_int4.py
Prints PROBE RESULT: SUCCESS or PROBE RESULT: FAILED, and why.
"""
import os
import sys

import numpy as np
import tensorflow as tf

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from tensorflow.lite.python import schema_py_generated as schema_fb  # noqa: E402

_PROBE_WINDOW = 32


def _tiny_model():
    inputs = tf.keras.Input(shape=(_PROBE_WINDOW, 1))
    x = tf.keras.layers.Conv1D(8, kernel_size=3, padding="same", activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(2, activation=None)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


def _representative_dataset():
    rng = np.random.RandomState(0)
    for _ in range(20):
        yield [rng.normal(size=(1, _PROBE_WINDOW, 1)).astype(np.float32)]


def _tensor_types(tflite_bytes):
    model = schema_fb.Model.GetRootAsModel(tflite_bytes, 0)
    subgraph = model.Subgraphs(0)
    types = []
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        types.append(t.Type())
    return types


def probe():
    model = _tiny_model()
    # A handful of throwaway fit steps so BN has real (non-random-init)
    # statistics to fold — irrelevant to int4 emission, just avoids
    # converting a model with degenerate BN variance.
    X = np.random.RandomState(0).normal(size=(64, _PROBE_WINDOW, 1)).astype(np.float32)
    y = np.random.RandomState(1).randint(0, 2, size=(64,))
    model.fit(X, y, epochs=1, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # This is the actual capability under test: does setting int4 weight
    # quantization on the converter change anything about the emitted
    # tensor types, or does it accept the option and silently emit int8
    # anyway? tf.lite's public API for this (as of 2.21) is
    # _experimental_low_bit_qat / target_spec.experimental_low_bit_...;
    # probe defensively since the attribute may not exist at all.
    used_int4_api = False
    for attr_name in ("experimental_low_bit_qat", "_experimental_low_bit_qat"):
        if hasattr(converter, attr_name):
            try:
                setattr(converter, attr_name, True)
                used_int4_api = True
            except Exception:
                pass

    try:
        tflite_bytes = converter.convert()
    except Exception as exc:  # noqa: BLE001 — this IS the probe's failure path
        print(f"PROBE RESULT: FAILED (converter.convert() raised: {exc!r})")
        print(f"used_int4_api={used_int4_api}")
        return False

    types = _tensor_types(tflite_bytes)
    type_names = {getattr(schema_fb.TensorType, n): n for n in dir(schema_fb.TensorType) if not n.startswith("_")}
    present = sorted({type_names.get(t, f"UNKNOWN({t})") for t in types})

    has_int4 = any(name == "INT4" for name in present)
    print(f"used_int4_api={used_int4_api}")
    print(f"tensor types present in converted model: {present}")
    if has_int4:
        print("PROBE RESULT: SUCCESS (INT4 tensors present)")
        return True
    else:
        print("PROBE RESULT: FAILED (no INT4 tensors emitted; converter fell back to INT8)")
        return False


if __name__ == "__main__":
    ok = probe()
    sys.exit(0 if ok else 1)
