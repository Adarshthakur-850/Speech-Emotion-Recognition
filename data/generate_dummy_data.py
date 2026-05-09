import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Feature size calculation:
# ZCR(1) + Chroma(12) + MFCC(40) + RMS(1) + Mel(128) = 182
FEATURE_SIZE = 182
CLASSES = ["Happy", "Sad", "Angry", "Fear", "Neutral", "Surprise"]
SAMPLES_PER_CLASS = 100

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_dummy_data():
    print("Generating dummy features...")
    X = []
    y = []

    for label in CLASSES:
        for _ in range(SAMPLES_PER_CLASS):
            # Generate random features
            # We add some bias to make them separable for demonstration
            bias = CLASSES.index(label) * 0.5
            features = np.random.rand(FEATURE_SIZE) + bias
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    
    # Encode labels
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Expand dims for CNN (samples, features, 1)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # Save data
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    
    # Save artifacts
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(DATA_DIR, "encoder.pkl"))
    
    print("Dummy data generated and saved.")

if __name__ == "__main__":
    generate_dummy_data()
