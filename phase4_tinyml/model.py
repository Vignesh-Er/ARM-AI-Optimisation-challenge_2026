import tensorflow as tf

def create_cnn_model(input_shape=(32, 1), n_classes=5):
    """Create a small 1D CNN for fault classification.
    
    Architecture (targeting <50KB quantized):
    - Conv1D(16, kernel=5, relu) + BatchNorm
    - Conv1D(32, kernel=3, relu) + BatchNorm
    - GlobalAveragePooling1D
    - Dense(32, relu) + Dropout(0.3)
    - Dense(n_classes, softmax)
    
    Returns:
        tf.keras.Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        
        tf.keras.layers.Conv1D(filters=16, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.GlobalAveragePooling1D(),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    
    return model
