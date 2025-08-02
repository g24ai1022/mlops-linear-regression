from sklearn.linear_model import LinearRegression
from utils import load_data, save_model, evaluate_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2, mse = evaluate_model(model, X_test, y_test)
    print(f"Training completed. R2 Score: {r2:.4f}, MSE: {mse:.4f}")

    model_path = save_model(model)
    print(f"Model saved to: {model_path}")