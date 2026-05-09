from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import tensorflow as tf
import joblib
import numpy as np
import librosa
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import extract_features

app = FastAPI(title="Speech Emotion Recognition API")

MODEL_PATH = "models/emotion_model.h5"
SCALER_PATH = "data/scaler.pkl"
ENCODER_PATH = "data/encoder.pkl"

model = None
scaler = None
encoder = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, encoder
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Model and artifacts loaded.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    temp_file = "temp.wav"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Load audio using librosa
        data, sr = librosa.load(temp_file, duration=2.5, offset=0.6)
        
        # Extract features
        features = extract_features.extract_features(data, sr)
        
        # Preprocess
        features = scaler.transform(features.reshape(1, -1))
        features = np.expand_dims(features, axis=2)
        
        # Predict
        prediction = model.predict(features)
        class_idx = np.argmax(prediction)
        
        label = encoder.categories_[0][class_idx]
        confidence = float(prediction[0][class_idx])
        
        return {"emotion": label, "confidence": confidence}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
