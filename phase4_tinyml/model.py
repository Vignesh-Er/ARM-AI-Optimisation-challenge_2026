# SPDX-License-Identifier: Apache-2.0
import tensorflow as tf


def create_tier2_model(input_shape=(32, 1), n_classes=5):
    """Tier-2: 5-class classifier, fully INT8 after quantization.

    CRITICAL: every conv block is Conv1D(activation=None) -> BatchNorm ->
    ReLU, not Conv1D(activation='relu') -> BatchNorm as the original model
    had it. BatchNorm can only be folded into the preceding convolution by
    the TFLite converter when it sits BEFORE the activation; with BN after
    a fused conv+relu, the converter has to emit a separate mul/add chain,
    for which CMSIS-NN has no kernel. See
    tools/verify_bn_fold.py, which asserts the converted model's op list
    contains no MUL/ADD between a CONV_2D and its activation.

    Architecture is frozen per docs/STATUS.md Phase 2 (Task 2.1):
        Conv1D(16, k=5, pad=same) -> BN -> ReLU   # INT8
        Conv1D(32, k=3, pad=same) -> BN -> ReLU   # INT8
        GlobalAveragePooling1D
        Dense(32) -> ReLU                         # INT8
        Dense(5)                                  # INT8
        softmax

    The trailing softmax is part of this Keras graph for training and for
    any human-readable probability output, but paci_infer_t2_s8() (Phase 2,
    Task 2.4) computes class_id/margin from the pre-softmax Dense(5) output
    directly — softmax is monotonic, so it changes neither the argmax class
    nor the *rank order* the margin is measured over, and the interface
    contract (paci_core.h) defines margin as "top logit minus runner-up",
    i.e. explicitly a pre-softmax quantity.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(16, kernel_size=5, padding="same", activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(32, activation=None)(x)
    x = tf.keras.layers.ReLU()(x)

    logits = tf.keras.layers.Dense(n_classes, activation=None, name="tier2_logits")(x)
    outputs = tf.keras.layers.Softmax()(logits)

    return tf.keras.Model(inputs, outputs, name="paci_tier2")


def create_tier1_model(input_shape=(32, 1)):
    """Tier-1: 2-class (normal vs anomalous) screen, mixed INT8/INT4.

    Architecture is frozen per docs/STATUS.md Phase 2 (Task 2.1):
        Conv1D(8, k=5, pad=same) -> BN -> ReLU    # INT8 (C_IN=1 blocks s4:
                                                   # a single input channel
                                                   # can't satisfy the packed
                                                   # int4 parity constraint)
        Conv1D(8, k=3, pad=same) -> BN -> ReLU    # INT4
        GlobalAveragePooling1D
        Dense(2)                                  # INT4

    No trailing softmax — unlike Tier-2, this model exists purely to
    produce a class_id/margin pair for the cascade's escalation decision
    (section 4), not a calibrated probability, so the graph ends at the raw
    2-logit Dense output.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(8, kernel_size=5, padding="same", activation=None, name="tier1_conv1_int8")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv1D(8, kernel_size=3, padding="same", activation=None, name="tier1_conv2_int4")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    logits = tf.keras.layers.Dense(2, activation=None, name="tier1_logits_int4")(x)

    return tf.keras.Model(inputs, logits, name="paci_tier1")
