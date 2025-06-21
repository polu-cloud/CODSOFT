import logging
import pandas as pd
from utils import load_artifact
import yaml

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Sample data to predict
sample = pd.DataFrame(
    [{"sepal_length": 6.1, "sepal_width": 2.9, "petal_length": 4.7, "petal_width": 1.4}]
)

print(f"Sample Data to predict:\n {sample}")

# Load model and encoder
logging.info("Loading model and encoder...")
model = load_artifact(config["model_path"])
encoder = load_artifact(config["encoder_path"])

# Predict
pred = model.predict(sample)
species = encoder.inverse_transform(pred)
logging.info(f"Predicted class: {species[0]}")
print(f"Prediction: {species[0]}")
