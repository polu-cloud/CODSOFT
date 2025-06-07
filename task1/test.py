import pandas as pd
import joblib
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DATASET = "Titanic-Dataset.csv"

logging.info("Loading model and data...")
model = joblib.load("titanic_model.pkl")
df = pd.read_csv(DATASET)

logging.info("Preprocessing test data...")
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
df.fillna({"Age": df["Age"].median()}, inplace=True)
df.fillna({"Embarked": df["Embarked"].mode()[0]}, inplace=True)

df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logging.info("Making predictions...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Accuracy: {accuracy:.4f}")
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

logging.info("Generating confusion matrix...")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
