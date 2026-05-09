import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
import os
import matplotlib.pyplot as plt

DATA_DIR = "data"
MODEL_PATH = "models/emotion_model.h5"

def train_model():
    print("Loading data...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    except FileNotFoundError:
        print("Data files not found. Please run 'python data/generate_dummy_data.py' first.")
        return

    input_shape = (X_train.shape[1], 1)
    num_classes = y_train.shape[1]

    print(f"Input Shape: {input_shape}, Classes: {num_classes}")

    # Build CNN Model for 1D Audio Features
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Save Model
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Plot
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join("training", "training_plot.png"))
    print("Training plot saved.")

if __name__ == "__main__":
    train_model()
