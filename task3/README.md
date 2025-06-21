# 🌼 Iris Flower Classification

This project builds a machine learning pipeline that classifies Iris flowers into:

-   Setosa
-   Versicolor
-   Virginica

based on sepal and petal measurements.

## 🔧 Features

-   Choose model (RandomForest, KNN, SVM) from `config.yaml`
-   CLI separation: `train.py` and `test.py`
-   Model + Encoder persistence
-   Logging throughout pipeline

## 🛠 How to Run (with UV package manager)

### 1. Install dependencies

```bash
uv sync
```

### 2. Train the model

```bash
uv run train.py
```

### 3. Test the model

```bash
uv run test.py
```

## 🛠 How to Run (with venv and pip)

### 0. Create a python virtual environment and activate it

```bash
python -m venv .venv
./.venv/Scripts/acitvate
```

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

### 3. Test the model

```bash
python test.py
```

---

## 🔄 Workflow Breakdown

### 🔹 Step 1: Configuration (`config.yaml`)

Define your parameters such as:

-   `model_type`: RandomForest, KNN, or SVM
-   `test_size`: Test dataset size (e.g. 0.2)
-   `random_state`: Seed value for reproducibility
-   `model_path`, `encoder_path`: Where to save model and encoder

---

### 🔹 Step 2: Training (`train.py`)

This script:

-   Loads and preprocesses the Iris dataset
-   Splits data into train/test
-   Trains the model defined in `config.yaml`
-   Evaluates accuracy and classification report
-   Saves trained model and label encoder in `artifacts/`

✅ Bonus: You can swap models by editing `model_type` in `config.yaml`.

---

### 🔹 Step 3: Prediction (`test.py`)

-   Loads a saved model and label encoder
-   Accepts a **hardcoded sample** (can be extended to user input)
-   Predicts the flower species

---

### 🔹 Step 4: Visualization (`train.py`)

We added Seaborn visualizations to help understand the feature distributions and class separability.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot pairplot colored by species
sns.pairplot(df, hue="species")
plt.suptitle("Iris Feature Pairwise Plots", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="YlGnBu")
plt.title("Feature Correlation Heatmap")
plt.show()

```
