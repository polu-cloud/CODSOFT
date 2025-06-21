import logging
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

from utils import load_data, preprocess, save_artifact

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_TYPE = config["model_type"]
TEST_SIZE = config["test_size"]
RANDOM_STATE = config["random_state"]

# Load and visualize dataset
logging.info("Loading dataset...")
df = load_data()

# ------- VISUALIZATION SECTION -------
logging.info("Generating visualizations...")

# Pairplot
sns.pairplot(df, hue="species")
plt.suptitle("Iris Feature Pairwise Plots", y=1.02)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ------- PREPROCESSING -------
logging.info("Preprocessing data...")
X, y, encoder = preprocess(df)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ------- MODEL SELECTION -------
logging.info(f"Training model: {MODEL_TYPE}")
if MODEL_TYPE == "RandomForest":
    model = RandomForestClassifier(random_state=RANDOM_STATE)
elif MODEL_TYPE == "KNN":
    model = KNeighborsClassifier()
elif MODEL_TYPE == "SVM":
    model = SVC()
else:
    raise ValueError("Invalid model type in config.yaml")

# Train the model
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save artifacts
save_artifact(model, config["model_path"])
save_artifact(encoder, config["encoder_path"])
logging.info("Model and encoder saved.")
