import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE
import os


def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # Create output folder if not exists
    os.makedirs("outputs/plots", exist_ok=True)

    # 💠 Plot class distribution before balancing
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Class")
    plt.title("Class Distribution Before Balancing")
    plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
    plt.ylabel("Count")
    plt.savefig("outputs/plots/class_distribution_before.png")
    plt.close()

    # Drop 'Time' and separate features and labels
    X = df.drop(["Class", "Time"], axis=1)
    y = df["Class"]

    # Correlation heatmap (sampled for performance)
    corr_sample = df.sample(n=10000, random_state=RANDOM_STATE)
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_sample.drop(columns="Time").corr(), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.savefig("outputs/plots/correlation_heatmap.png")
    plt.close()

    # Distribution of transaction amount
    plt.figure(figsize=(6, 4))
    sns.histplot(df["Amount"], bins=50, kde=True)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.savefig("outputs/plots/amount_distribution.png")
    plt.close()

    # Balance the dataset using undersampling
    df_combined = pd.concat([X, y], axis=1)
    df_majority = df_combined[df_combined["Class"] == 0]
    df_minority = df_combined[df_combined["Class"] == 1]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority) * 5,
        random_state=RANDOM_STATE,
    )

    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    # 💠 Plot class distribution after balancing
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_balanced, x="Class")
    plt.title("Class Distribution After Balancing")
    plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
    plt.ylabel("Count")
    plt.savefig("outputs/plots/class_distribution_after.png")
    plt.close()

    # Split and scale
    X_bal = df_balanced.drop("Class", axis=1)
    y_bal = df_balanced["Class"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)

    return train_test_split(
        X_scaled, y_bal, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
