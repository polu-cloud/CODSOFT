import argparse
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def train_model(X, y):
    # Initialize base Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # Define hyperparameter grid to search
    params = {
        "n_estimators": [50, 100],  # Number of trees in the forest
        "max_depth": [None, 10, 20],  # Maximum depth of each tree
    }

    # Setup GridSearchCV to find best combination using 3-fold cross-validation
    gs = GridSearchCV(
        rf,
        params,
        cv=3,
        scoring="neg_root_mean_squared_error",  # Evaluation metric (negated RMSE)
        n_jobs=-1,  # Use all CPU cores for parallel processing
    )

    # Train the model using all combinations
    gs.fit(X, y)

    # Return the best model found during grid search
    return gs.best_estimator_


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train rating regression")
    parser.add_argument(
        "--data", type=Path, required=True, help="features CSV (with Rating)"
    )
    parser.add_argument(
        "--model-out", type=Path, required=True, help="where to save .pkl"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y = df["Rating"]
    X = df.drop(columns=["Rating"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    preds = model.predict(X_val)
    print("Validation RMSE:", root_mean_squared_error(y_val, preds))
    print("Validation MAE: ", mean_absolute_error(y_val, preds))
    print("Validation R2:  ", r2_score(y_val, preds))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
