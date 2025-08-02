import numpy as np
import joblib
from utils import load_model, load_data, evaluate_model

if __name__ == "__main__":
    model = load_model()
    X_train, X_test, y_train, y_test = load_data()

    # Save raw params
    raw_params = {"coef_": model.coef_, "intercept_": model.intercept_}
    joblib.dump(raw_params, "artifacts/unquant_params.joblib")
    print("Raw parameters saved.")

    # Use int16 for better precision
    scale_factor = 1000  # adjust precision
    coef_q = np.round(model.coef_ * scale_factor).astype(np.int16)
    intercept_q = np.round(model.intercept_ * scale_factor).astype(np.int32)
    print("coef_q ",coef_q)
    print("intercept_q ",intercept_q)

    quant_params = {
        "coef_q": coef_q,
        "intercept_q": intercept_q,
        "scale_factor": scale_factor
    }
    joblib.dump(quant_params, "artifacts/quant_params.joblib")
    print("Quantized parameters saved.")

    # De-quantize for inference
    coef_dq = coef_q.astype(np.float32) / scale_factor
    intercept_dq = intercept_q.astype(np.float32) / scale_factor

    # Manual prediction using quantized weights
    y_pred_quant = X_test @ coef_dq + intercept_dq
    r2_original = evaluate_model(model, X_test, y_test)[0]
    ss_res = np.sum((y_test - y_pred_quant)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    r2_quant = 1 - (ss_res / ss_tot)

    print(f"Original R2: {r2_original:.4f}")
    print(f"Quantized R2: {r2_quant:.4f}")
