import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.config import MODEL_PATH


def train_model(X_train, y_train, model_type="logreg"):
    logging.info(f"Training model: {model_type}")
    if model_type == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Invalid model type")

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    return model
