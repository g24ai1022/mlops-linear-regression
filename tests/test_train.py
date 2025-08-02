import joblib
import os
from src.utils import load_data, save_model, load_model, evaluate_model
from src.train import train_and_save_model
from sklearn.linear_model import LinearRegression

def test_data_loading():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0

def test_model_training():
    model, r2 = train_and_save_model()
    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")
    assert r2 > 0.4  # Example threshold

def test_model_file_created():
    path = "artifacts/test_model.joblib"
    assert os.path.exists(path)

