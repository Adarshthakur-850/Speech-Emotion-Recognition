import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
import os

API_URL = "http://127.0.0.1:8000/predict"

st.title("🎤 Speech Emotion Recognition")

# Mode Selection
mode = st.radio("Select Mode", ["Upload Audio", "Live Recording"])

if mode == "Upload Audio":
    uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("Analyze Audio"):
            files = {"file": uploaded_file.getvalue()}
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    res = response.json()
                    st.success(f"Detected Emotion: **{res['emotion']}**")
                    st.info(f"Confidence: {res['confidence']:.2f}")
                else:
                    st.error("Error from API")
            except Exception as e:
                st.error(f"Connection Error: {e}")

elif mode == "Live Recording":
    duration = st.slider("Recording Duration (seconds)", 2, 5, 3)
    if st.button("Record & Analyze"):
        fs = 22050
        st.write("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        st.write("Recording finished.")
        
        # Save temp file
        wav.write("temp_rec.wav", fs, recording)
        
        # Send to API
        with open("temp_rec.wav", "rb") as f:
            files = {"file": f}
            try:
                response = requests.post(API_URL, files=files)
                if response.status_code == 200:
                    res = response.json()
                    st.success(f"Detected Emotion: **{res['emotion']}**")
                    st.info(f"Confidence: {res['confidence']:.2f}")
                else:
                    st.error("Error from API")
            except Exception as e:
                st.error(f"Connection Error: {e}")
