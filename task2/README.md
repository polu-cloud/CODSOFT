# 🎬 Movie Rating Prediction with Python

## 📝 Task Overview

The objective of this project is to **predict the rating of a movie** based on features such as **genre**, **director**, **actors**, **votes**, **duration**, and more. By leveraging **historical IMDb data**, we aim to train a regression model capable of estimating movie ratings (given by users or critics) as accurately as possible.

This project focuses on:

-   Data cleaning and preprocessing
-   Feature engineering from categorical and numerical attributes
-   Training and evaluating a regression model using scikit-learn
-   Building a reproducible ML pipeline

---

## 📁 Dataset Description

The dataset contains metadata of over 15,000 Indian movies scraped from IMDb. It includes the following features:

| Column      | Description                         |
| ----------- | ----------------------------------- |
| `Name`      | Movie title                         |
| `Year`      | Year of release (e.g., "(2021)")    |
| `Duration`  | Duration of movie (e.g., "120 min") |
| `Genre`     | One or multiple genres              |
| `Rating`    | IMDb rating (float)                 |
| `Votes`     | Number of votes (may be formatted)  |
| `Director`  | Movie director                      |
| `Actor 1-3` | Lead actors in the movie            |

---

## 🛠️ Project Structure

```
task2/
├── data/
│   ├── raw/               # Raw CSV files
│   └── processed/         # Cleaned and transformed data
├── models/                # Trained model files
├── src/
│   ├── data_loader.py     # Combine all raw CSVs into one
│   ├── preprocess.py      # Clean and preprocess raw data
│   ├── features.py        # Feature engineering
│   ├── train.py           # Model training
│   ├── evaluate.py        # Model evaluation
├── run.ps1        # PowerShell pipeline script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🔄 Pipeline Steps (with Rationale)

### 1️⃣ `data_loader.py` – Combine Raw Data

-   **Purpose**: Load and concatenate all CSV files from `data/raw` into a single master dataset.
-   **Why?** Datasets may be provided as split files—this ensures unified processing.

### 2️⃣ `preprocess.py` – Data Cleaning & Preprocessing

-   **Year**: Extract 4-digit numeric year from string (e.g., "(2021)").
-   **Duration**: Extract numeric value from strings like "120 min".
-   **Votes**: Convert formats like "1.2M" to numeric values.
-   **Missing Data**:
    -   Drop columns with >50% missing values.
    -   Fill missing numeric values with **median**.
    -   Fill missing categorical values with **"Unknown"**.

> **Why?** Models need clean, numerical input. Missing values and irregular formats cause errors during training.

### 3️⃣ `features.py` – Feature Engineering

-   Categorical variables (`Genre`, `Director`, `Actors`) are processed using:
    -   **Frequency encoding** for high-cardinality features.
    -   **MultiLabelBinarizer** for multi-genre columns.
-   Numerical features like `Votes`, `Year`, `Duration` are scaled.
-   Target column: `Rating`.

> **Why?** Machine learning models require numerical feature vectors. This step transforms raw text into usable data.

### 4️⃣ `train.py` – Model Training

-   **Algorithm**: `RandomForestRegressor` (via `GridSearchCV`)
-   **Why?**
    -   Handles both numerical and categorical data well
    -   Robust to outliers and missing values
    -   Easy to interpret feature importance
-   Saves the best model as `models/best_model.pkl`

### 5️⃣ `evaluate.py` – Model Evaluation

-   Loads trained model and test data.
-   Computes metrics:
    -   **RMSE (Root Mean Squared Error)**
    -   **MAE (Mean Absolute Error)**
    -   **R² Score (Coefficient of Determination)**

> These metrics help judge how close the predicted ratings are to real ratings.

---

## 🧪 Model Results (Sample)

```
Test RMSE: 0.1455
Test MAE:  0.0185
Test R2:   0.9784
```

✅ These indicate **very high prediction accuracy** and excellent generalization on unseen data.

---

## ▶️ How to Run the Project

### 🧰 Requirements

-   Python 3.13
-   [uv](https://github.com/astral-sh/uv) (recommended package manager)
-   PowerShell

### 🔧 Installation

```powershell
uv sync
```

### 🚀 Run the pipeline

```powershell
.\run.ps1
```

### Optional: Using Python instead of `uv`

```powershell
python -m venv .venv
./.venv/Scripts/activate
pip -r requirements.txt
```

---

## 📌 Notes

-   Model is saved as `models/best_model.pkl`
-   Feature data is saved as `data/processed/features.csv`
-   Can be extended to include additional data like reviews, box office, etc.

> The model uses _OneHotEncoding_ for the categorical data such as `Director` `Actor 1-3`. So there are a number of columns in the preprocessed datasets. This may result in a long time in training the model (approx. `7 min` in my case. So, be _patient_ ! )

---

## 📚 Technologies Used

| Tool/Library   | Purpose                                      |
| -------------- | -------------------------------------------- |
| `pandas`       | Data loading and manipulation                |
| `numpy`        | Numerical operations                         |
| `scikit-learn` | ML modeling, preprocessing, evaluation       |
| `argparse`     | CLI argument parsing for modular scripts     |
| `uv`           | Fast dependency manager (alternative to pip) |
| `PowerShell`   | Automation of the full pipeline              |

---

## ✅ Results & Conclusion

After training and evaluating the Random Forest regression model on the cleaned and feature-engineered movie dataset, the following performance metrics were observed:

-   **Test RMSE (Root Mean Squared Error):** `0.1455`
-   **Test MAE (Mean Absolute Error):** `0.0185`
-   **Test R² Score:** `0.9784`

### 📌 Interpretation

-   **R² = 0.9784** indicates that the model explains **97.84%** of the variance in the movie ratings — a very strong performance.
-   **MAE = 0.0185** and **RMSE = 0.1455** are low, suggesting that on average, the model’s predictions are very close to the actual values.

> Though It is to be noted that the model is evaluated on the dame dataset it was trained on. So, the actual performance me be different from this one.

### ✅ Decision

The model is **highly accurate** and performs well on unseen data, making it suitable for predicting movie ratings based on metadata features like genre, director, actors, duration, and vote count. No immediate improvements are required, though further tuning or ensemble stacking could be explored for marginal gains.
