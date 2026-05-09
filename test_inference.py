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

def test_inference_pipeline():
    print("Testing Inference Pipeline...")
    
    # 1. Load Artifacts
    if not os.path.exists(MODEL_PATH):
        print(f"FAILED: Model not found at {MODEL_PATH}")
        return
        
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"FAILED to load artifacts: {e}")
        return

    # 2. Simulate Audio Data
    # 2.5 seconds at 22050 Hz = 55125 samples
    fs = 22050
    seconds = 2.5
    dummy_audio = np.random.uniform(-1, 1, int(seconds * fs))
    
    print(f"Generated dummy audio: {dummy_audio.shape}")

    # 3. Extract Features
    try:
        features = extract_features.extract_features(dummy_audio, fs)
        print(f"Extracted features shape: {features.shape}") # Should be (182,)
    except Exception as e:
        print(f"FAILED feature extraction: {e}")
        return

    # 4. Preprocess (Scale & Reshape)
    try:
        # Scale
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Reshape for CNN (samples, features, 1)
        features_cnn = np.expand_dims(features_scaled, axis=2)
        print(f"Input shape for model: {features_cnn.shape}")
    except Exception as e:
        print(f"FAILED preprocessing: {e}")
        return

    # 5. Predict
    try:
        prediction = model.predict(features_cnn, verbose=0)
        class_idx = np.argmax(prediction)
        
        # Decode label
        label = encoder.categories_[0][class_idx]
        confidence = prediction[0][class_idx]
        
        print(f"SUCCESS: Prediction ran successfully.")
        print(f"Predicted Emotion: {label} (Confidence: {confidence:.2f})")
    except Exception as e:
        print(f"FAILED prediction: {e}")

if __name__ == "__main__":
    test_inference_pipeline()
