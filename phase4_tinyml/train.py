import tensorflow as tf
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase4_tinyml.model import create_cnn_model

import numpy as np

def train_and_quantize(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train the CNN and quantize to int8 TFLite.
    
    1. Train with Adam optimizer, sparse categorical crossentropy
    2. Save best model (by val_accuracy)
    3. Convert to int8 TFLite with representative dataset
    4. Save TFLite model to outputs/models/
    5. Report model sizes (Keras vs TFLite)
    
    Returns:
        model: trained Keras model
        tflite_model_path: path to quantized model
        history: training history
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    n_classes = config.N_CLASSES
    
    model = create_cnn_model(input_shape=input_shape, n_classes=n_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.CNN_LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save best model
    keras_model_path = os.path.join(config.MODELS_DIR, 'best_cnn.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        keras_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop],
        verbose=0
    )
    
    print("Quantizing model to INT8 TFLite...")
    
    def representative_dataset():
        # Yield a few samples for calibration
        for i in range(100):
            yield [X_train[i:i+1].astype(np.float32)]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    # Ensure full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    tflite_model_path = os.path.join(config.MODELS_DIR, 'cnn_quantized.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_quant_model)
        
    # Sizes
    if os.path.exists(keras_model_path):
        keras_size = os.path.getsize(keras_model_path) / 1024
        print(f"Keras Model Size: {keras_size:.2f} KB")
    
    tflite_size = os.path.getsize(tflite_model_path) / 1024
    print(f"TFLite INT8 Model Size: {tflite_size:.2f} KB")
    
    return model, tflite_model_path, history
