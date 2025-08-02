import joblib
import numpy as np

def quantize_weights(weights):
    return np.clip((weights * 10).astype(np.uint8), 0, 255)

def dequantize_weights(qweights):
    return qweights.astype(np.float32) / 10.0

def main():
    model = joblib.load("models/model.joblib")
    raw_params = {
        "coef": model.coef_,
        "intercept": model.intercept_
    }
    joblib.dump(raw_params, "models/unquant_params.joblib")

    quantized = {
        "coef": quantize_weights(model.coef_),
        "intercept": quantize_weights(np.array([model.intercept_]))
    }
    joblib.dump(quantized, "models/quant_params.joblib")

if __name__ == "__main__":
    main()

