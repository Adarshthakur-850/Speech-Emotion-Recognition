# Speech Emotion Recognition (SER) System

A real-time Deep Learning system that detects emotions from audio input (Microphone or File).

## Features
- **Emotion Classes**: Happy, Sad, Angry, Fear, Neutral, Surprise.
- **Deep Learning**: Uses a 1D CNN trained on audio features (MFCC, Mel Spectrogram, etc.).
- **Real-time**: Live microphone emotion detection (`live/mic_emotion.py`).
- **Web Interface**: Streamlit UI for recording and analyzing audio.

## Project Structure
```
Speech Emotion Recognition/
│
├── data/generate_dummy_data.py # Synthetic training data
├── features/extract_features.py # Audio feature extraction logic
├── training/train_model.py      # Model training script
├── live/mic_emotion.py          # Real-time microphone inference
├── api/app.py                   # FastAPI backend
├── ui/streamlit_app.py          # Frontend UI
└── requirements.txt
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need `portaudio` installed for `pyaudio`/`sounddevice`.*

2.  **Generate Data** (for demo purpose):
    ```bash
    python data/generate_dummy_data.py
    ```

3.  **Train Model**:
    ```bash
    python training/train_model.py
    ```

## Running the Application

### 1. Live Microphone Detection (CLI)
```bash
python live/mic_emotion.py
```

### 2. Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

### 3. API
```bash
uvicorn api.app:app --reload
```
