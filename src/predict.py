if __name__ == "__main__":
    model = load_model()
    _, X_test, _, _ = load_data()

    predictions = model.predict(X_test[:5])
    print("Sample Predictions on first 5 test rows:")
    for i, pred in enumerate(predictions, start=1):
        print(f"Row {i}: {pred:.4f}")
