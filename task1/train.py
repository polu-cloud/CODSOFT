import pandas as pd
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DATASET = "Titanic-Dataset.csv"


logging.info(f"Loading dataset {DATASET}...")
df = pd.read_csv(DATASET)

logging.info("Preprocessing data...")
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)

df.fillna({"Age": df["Age"].median()}, inplace=True)
df.fillna({"Embarked": df["Embarked"].mode()[0]}, inplace=True)


le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

logging.info("Splitting dataset into train and test sets...")
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logging.info("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

logging.info("Saving trained model and label encoder...")
joblib.dump(model, "titanic_model.pkl")
joblib.dump(le, "label_encoder.pkl")

logging.info("Model training complete and saved.")
