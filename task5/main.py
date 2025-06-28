import argparse
from src.utils import setup_logging
from src.preprocess import load_and_preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def main(args):
    setup_logging()
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = train_model(X_train, y_train, model_type=args.model)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "rf"],
        help="Model type to use: logreg or rf",
    )
    args = parser.parse_args()
    main(args)
