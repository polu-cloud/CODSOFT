from sklearn.metrics import classification_report, confusion_matrix
import logging


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    logging.info("Classification Report:\n" + str(report))
    logging.info("Confusion Matrix:\n" + str(cm))
    print(report)
    print("Confusion Matrix:\n", cm)
