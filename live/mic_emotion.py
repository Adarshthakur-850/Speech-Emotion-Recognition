import sounddevice as sd
import numpy as np
import tensorflow as tf
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import extract_features

MODEL_PATH = "models/emotion_model.h5"
SCALER_PATH = "data/scaler.pkl"
ENCODER_PATH = "data/encoder.pkl"

def live_prediction():
    print("Loading model and artifacts...")
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    
    fs = 22050  # Sample rate
    seconds = 2.5  # Duration of recording

    print("Listening... (Press Ctrl+C to stop)")
    
    try:
        while True:
            print("Recording...", end="\r")
            myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            sd.wait()  # Wait until recording is finished
            
            # Extract features from buffer (flatten to 1D)
            data = myrecording.flatten()
            
            # We need to ensure we use the same feature extraction logic as training
            # Use the extract_features function
            features = extract_features.extract_features(data, fs)
            
            # Scale
            features = scaler.transform(features.reshape(1, -1))
            
            # Reshape for CNN (samples, features, 1)
            features = np.expand_dims(features, axis=2)
            
            # Predict
            prediction = model.predict(features, verbose=0)
            class_idx = np.argmax(prediction)
            
            # Decode label
            # encoder.categories_[0] holds the classes
            label = encoder.categories_[0][class_idx]
            confidence = prediction[0][class_idx]
            
            print(f"Detected: {label} ({confidence:.2f})      ")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    live_prediction()
