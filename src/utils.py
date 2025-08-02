import os
import joblib

MODEL_DIR = "artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model(model, filename="model.joblib"):
    filepath = os.path.join(MODEL_DIR, filename)
    joblib.dump(model, filepath)
    return filepath

def load_model(filename="model.joblib"):
    filepath = os.path.join(MODEL_DIR, filename)
    return joblib.load(filepath)
