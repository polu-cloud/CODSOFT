import argparse
import pandas as pd
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate trained model")
    parser.add_argument("--model-path", type=str, required=True, help="model .pkl")
    parser.add_argument(
        "--test-data", type=str, required=True, help="features CSV (with Rating)"
    )
    args = parser.parse_args()

    # Load test data
    df = pd.read_csv(args.test_data)
    y_true = df["Rating"]  # Extract true ratings
    X_test = df.drop(columns=["Rating"])  # Drop target to get features only

    # Load the trained model
    model = joblib.load(args.model_path)

    # Predict on test set
    preds = model.predict(X_test)

    print(f"Test RMSE: {root_mean_squared_error(y_true, preds):.4f}")
    print(f"Test MAE:  {mean_absolute_error(y_true, preds):.4f}")
    print(f"Test R2:   {r2_score(y_true, preds):.4f}")
