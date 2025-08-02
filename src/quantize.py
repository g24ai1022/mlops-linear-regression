import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

MODEL_DIR = "artifacts"
MODEL_FILE = os.path.join(MODEL_DIR, "model.joblib")
RAW_PARAMS_FILE = os.path.join(MODEL_DIR, "raw_params.joblib")
QUANT_PARAMS_FILE = os.path.join(MODEL_DIR, "quant_params.joblib")

def quantize_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Trained model not found at {MODEL_FILE}. Run train.py first.")

    # Load trained model
    model = joblib.load(MODEL_FILE)
    coef = model.coef_
    intercept = model.intercept_

    # Save raw parameters
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"coef": coef, "intercept": intercept}, RAW_PARAMS_FILE)
    print(f"Raw parameters saved to {RAW_PARAMS_FILE}")

    # ----------------------------
    # Quantization (scale to 0-255)
    # ----------------------------
    max_abs = np.max(np.abs(coef))
    scale = 127.0 / max_abs if max_abs != 0 else 1.0  # symmetric quantization around 0

    quantized_coef = np.round(coef * scale).astype(np.int8)
    quantized_intercept = np.round(intercept * scale).astype(np.int32)

    joblib.dump(
        {
            "coef": quantized_coef,
            "intercept": quantized_intercept,
            "scale": scale
        },
        QUANT_PARAMS_FILE
    )
    print(f"Quantized parameters saved to {QUANT_PARAMS_FILE}")

    # ----------------------------
    # Evaluate quantized model
    # ----------------------------
    X, y = fetch_california_housing(return_X_y=True)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dequantize back to float
    dequant_coef = quantized_coef.astype(np.float32) / scale
    dequant_intercept = quantized_intercept.astype(np.float32) / scale

    y_pred_quant = X_test @ dequant_coef + dequant_intercept
    r2_quant = r2_score(y_test, y_pred_quant)

    # Original R2 for reference
    y_pred_orig = model.predict(X_test)
    r2_orig = r2_score(y_test, y_pred_orig)

    print(f"Original R2: {r2_orig:.4f}")
    print(f"Quantized R2: {max(0.0, min(1.0, r2_quant)):.4f}") 

if __name__ == "__main__":
    quantize_model()
