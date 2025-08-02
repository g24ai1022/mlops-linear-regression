from sklearn.linear_model import LinearRegression
from utils import load_data, save_model, evaluate_model

def train_and_save_model(filename="model.joblib"):
    X_train, X_test, y_train, y_test = load_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2, mse = evaluate_model(model, X_test, y_test)
    print(f"Training completed. R2 Score: {r2:.4f}, MSE: {mse:.4f}")

    model_path = save_model(model, filename)
    print(f"Model saved to: {model_path}")

    return model, r2

if __name__ == "__main__":
    train_and_save_model()
