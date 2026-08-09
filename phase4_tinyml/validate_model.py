import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1_physics.physics_model import PhysicsModel
from phase4_tinyml.dataset import generate_cnn_dataset
from phase4_tinyml.train import train_and_quantize

def evaluate_model(model, X_test, y_test):
    # Keras model evaluation
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES[:np.max(y_test)+1], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return y_pred, report, cm

def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'phase4_training_history.png'))
    plt.close()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES[:cm.shape[0]], 
                yticklabels=config.CLASS_NAMES[:cm.shape[0]])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, 'phase4_confusion_matrix.png'))
    plt.close()

def main():
    print("Generating CNN dataset...")
    physics_model = PhysicsModel()
    X_train, X_val, X_test, y_train, y_val, y_test = generate_cnn_dataset(
        physics_model, n_scenarios=20, n_steps_per=500, seed=config.SEED
    )
    
    print(f"Dataset sizes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    print("\nTraining and Quantizing...")
    model, tflite_path, history = train_and_quantize(
        X_train, y_train, X_val, y_val, 
        epochs=config.CNN_EPOCHS, batch_size=config.CNN_BATCH_SIZE
    )
    
    print("\nEvaluating Keras model on test set...")
    y_pred, report, cm = evaluate_model(model, X_test, y_test)
    
    plot_history(history)
    plot_confusion_matrix(cm)
    
    print("\n--- CNN Evaluation Summary ---")
    print(f"Overall Accuracy: {report['accuracy']*100:.2f}%")
    
    print("\nPer-Class Metrics:")
    for class_id in range(len(cm)):
        class_name = config.CLASS_NAMES[class_id]
        if class_name in report:
            prec = report[class_name]['precision']
            rec = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            print(f"  {class_name:20s}: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
            
    # TFLite size
    tflite_size = os.path.getsize(tflite_path) / 1024
    print(f"\nFinal TFLite Model Size: {tflite_size:.2f} KB")

if __name__ == '__main__':
    main()
