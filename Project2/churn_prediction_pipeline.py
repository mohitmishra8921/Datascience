"""
Customer Churn Prediction - Production Pipeline
=================================================
End-to-end pipeline: data loading, preprocessing, model training,
evaluation, and final prediction export.

Author: <your name>
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42
DATA_PATH = "customer_churn.csv"
MODEL_OUTPUT_PATH = "churn_model.pkl"
SCALER_OUTPUT_PATH = "scaler.pkl"
TEST_INPUT_CSV = "test_input.csv"
TEST_OUTPUT_CSV = "test_predictions.csv"

BINARY_COLS = ["Partner", "Dependents", "Phoneservice", "Paperlessbilling", "Churn"]
MULTI_CATEGORY_COLS = [
    "Multiplelines", "Internetservice", "Onlinesecurity", "Onlinebackup",
    "Deviceprotection", "Techsupport", "Streamingtv", "Streamingmovies",
    "Contract", "Paymentmethod"
]
NUMERICAL_COLS = ["Tenure", "Monthlycharges", "Totalcharges"]


# ----------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load raw churn dataset and standardize column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.capitalize()
    return df


# ----------------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the raw dataframe:
    - Drop non-predictive ID column
    - Label-encode binary Yes/No columns
    - One-hot encode multi-category columns
    """
    df = df.copy()

    # Drop unique identifier - carries no predictive signal
    if "Customerid" in df.columns:
        df = df.drop("Customerid", axis=1)

    # Binary encoding
    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # One-hot encoding (drop_first avoids the dummy variable trap)
    df = pd.get_dummies(df, columns=MULTI_CATEGORY_COLS, drop_first=True)

    return df


def split_and_scale(df: pd.DataFrame):
    """
    Split into train/test (stratified on target) and scale numerical
    columns using a scaler fit ONLY on the training set (avoids leakage).
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])

    return X_train, X_test, y_train, y_test, scaler


# ----------------------------------------------------------------------
# 3. MODEL TRAINING
# ----------------------------------------------------------------------
def train_models(X_train, y_train):
    """Train candidate models and return them in a dict."""
    models = {}

    models["logistic_regression"] = LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000
    ).fit(X_train, y_train)

    models["random_forest"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    ).fit(X_train, y_train)

    models["xgboost"] = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    ).fit(X_train, y_train)

    return models


# ----------------------------------------------------------------------
# 4. EVALUATION
# ----------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Compute standard classification metrics for a trained model."""
    y_pred = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k.capitalize()}: {v:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return metrics


def select_best_model(models: dict, X_test, y_test):
    """Evaluate all models and return the one with the highest F1 score."""
    results = [evaluate_model(m, X_test, y_test, name) for name, m in models.items()]
    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    return models[best_name], best_name, results_df


# ----------------------------------------------------------------------
# 5. FINAL TEST SIMULATION (input/output CSV pair)
# ----------------------------------------------------------------------
def export_test_simulation(model, X_test, y_test, scaler):
    """
    Save two CSVs simulating a real-world deployment check:
    - test_input.csv: features only, as a model would receive them
    - test_predictions.csv: model predictions vs actual ground truth
    """
    X_test.to_csv(TEST_INPUT_CSV, index=False)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    output_df = pd.DataFrame({
        "Actual_Churn": y_test.values,
        "Predicted_Churn": predictions,
        "Churn_Probability": np.round(probabilities, 4),
        "Correct_Prediction": (y_test.values == predictions)
    })
    output_df.to_csv(TEST_OUTPUT_CSV, index=False)

    print(f"\nSaved: {TEST_INPUT_CSV} ({X_test.shape[0]} rows)")
    print(f"Saved: {TEST_OUTPUT_CSV} ({output_df.shape[0]} rows)")
    print(f"Simulation accuracy check: {output_df['Correct_Prediction'].mean():.4f}")


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    print("Loading data...")
    df_raw = load_data(DATA_PATH)

    print("Preprocessing...")
    df_processed = preprocess_data(df_raw)

    print("Splitting and scaling...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_processed)

    print("Training models...")
    models = train_models(X_train, y_train)

    print("Evaluating models...")
    best_model, best_name, results_df = select_best_model(models, X_test, y_test)
    print(f"\nBest model selected: {best_name}")

    # Feature importance (only for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        print("\nTop 10 important features:")
        print(importance_df.head(10).to_string(index=False))

    print("\nSaving model and scaler...")
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)

    print("\nExporting test simulation CSVs...")
    export_test_simulation(best_model, X_test, y_test, scaler)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()