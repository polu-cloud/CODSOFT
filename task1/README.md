# 🚢 Titanic Survival Prediction

This project uses the Titanic dataset to build a **Random Forest Classifier** that predicts whether a passenger survived the Titanic disaster based on features like age, sex, fare, and passenger class.

---

## 📁 Project Structure

```
label_encoder.pkl   ## Saved label encoder for categorical features
main.ipynb      ## Initial code
test.py ## Loads the model and evaluates it
titanic_model.pkl   ## Saved random forest model
Titanic-Dataset.csv ## Input dataset
train.py ## Trains the model and saves it
```

---

## 📌 Requirements

Install dependencies using:

✅ With uv (faster dependency manager)

```bash
uv sync #(with uv)
```

✅ With Python Venv directly

```bash
pip -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## ⚙️ How to Run

✅ With uv (faster dependency manager)

1. Train the model
    ```bash
    uv run train.py
    ```
2. Test the model
    ```bash
    uv run test.py
    ```

✅ With Python virtual environment directly
First make sure to activate the virtual environment

1. Train the model:
    ```bash
    python train.py
    ```
2. Test the model
    ```bash
    python test.py
    ```

## 📊 Sample Output

```bash
└ $ uv run .\train.py
2025-06-08 01:22:58,522 - INFO - Loading dataset Titanic-Dataset.csv...
2025-06-08 01:22:58,525 - INFO - Preprocessing data...
2025-06-08 01:22:58,528 - INFO - Splitting dataset into train and test sets...
2025-06-08 01:22:58,529 - INFO - Training Random Forest model...
2025-06-08 01:22:58,639 - INFO - Saving trained model and label encoder...
2025-06-08 01:22:58,664 - INFO - Model training complete and saved.


└ $ uv run .\test.py
2025-06-08 01:23:18,101 - INFO - Loading model and data...
2025-06-08 01:23:18,233 - INFO - Preprocessing test data...
2025-06-08 01:23:18,237 - INFO - Making predictions...
2025-06-08 01:23:18,243 - INFO - Accuracy: 0.8212
Accuracy: 0.8212290502793296

Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.88      0.85       105
           1       0.81      0.74      0.77        74

    accuracy                           0.82       179
   macro avg       0.82      0.81      0.81       179
weighted avg       0.82      0.82      0.82       179

2025-06-08 01:23:18,250 - INFO - Generating confusion matrix...
```
