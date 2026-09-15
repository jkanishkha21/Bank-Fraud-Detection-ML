"""
Train and evaluate the Bank Fraud Detection model.

Model:
- OneHotEncoder for categorical variables
- StandardScaler for numeric variables
- RandomForestClassifier with class weighting
- Threshold selection using the validation set
- Evaluation using precision, recall, F1 and ROC-AUC

Outputs:
- models/fraud_pipeline.joblib
- models/model_metadata.json
- reports/metrics.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = "data/transactions.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"
RANDOM_STATE = 42

TARGET = "is_fraud"

NUMERIC_FEATURES = [
    "amount",
    "hour",
    "device_risk_score",
    "ip_risk_score",
    "account_age_days",
    "previous_transactions",
    "international",
]

CATEGORICAL_FEATURES = [
    "transaction_type",
    "merchant_category",
    "country",
]


def choose_threshold(y_true, probabilities):
    """Choose a threshold that gives a useful recall/F1 trade-off."""
    best_threshold = 0.50
    best_score = -1

    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        recall = recall_score(y_true, predictions, zero_division=0)
        precision = precision_score(y_true, predictions, zero_division=0)

        # Favor recall while still rewarding precision.
        score = 0.7 * recall + 0.3 * precision

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python generate_dataset.py` first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train_sub, y_train_sub)

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]
    threshold = choose_threshold(y_val, val_probabilities)

    # Refit the same pipeline on the full training set after threshold selection.
    pipeline.fit(X_train, y_train)

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, test_predictions, zero_division=0)),
        "fraud_rate": float(y.mean()),
        "selected_threshold": threshold,
        "confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            test_predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    joblib.dump(pipeline, os.path.join(MODEL_DIR, "fraud_pipeline.joblib"))

    metadata = {
        "threshold": threshold,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "target": TARGET,
        "model": "RandomForestClassifier",
        "random_state": RANDOM_STATE,
    }

    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel training completed.")
    print(f"Fraud rate: {metrics['fraud_rate']:.2%}")
    print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"Threshold:   {threshold:.2f}")
    print(f"Model saved to {MODEL_DIR}/fraud_pipeline.joblib")
    print(f"Metrics saved to {REPORT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
