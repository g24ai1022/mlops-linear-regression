from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import os

def load_data():
    X, y = fetch_california_housing(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def git_commit_model(model_path):
    try:
        # Stage the model file
        subprocess.run(["git", "add", model_path], check=True)

        # Commit with a message
        commit_message = "Add trained model file"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # Push to the current branch (assumes credentials are setup)
        subprocess.run(["git", "push"], check=True)

        print("Model committed and pushed to git successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")

def train_and_save_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(os.getcwd(), "models", "model.joblib")
    X_train, X_test, y_train, y_test = load_data()
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f"R2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    return model, r2

if __name__ == "__main__":
    train_and_save_model()
