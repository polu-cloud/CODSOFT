import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import yaml

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

DATA_PATH = config["data_path"]


def load_data():
    return pd.read_csv(DATA_PATH)


def preprocess(df):
    le = LabelEncoder()
    df["species"] = le.fit_transform(df["species"])
    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y, le


def save_artifact(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path):
    return joblib.load(path)
