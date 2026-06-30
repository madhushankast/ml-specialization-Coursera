import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt


def load_data():
    """Load processed weather dataset splits and return train/test features and binary targets.
    Returns X_train, y_train, X_test, y_test as pandas DataFrames/Series with targets encoded as 0 (No) and 1 (Yes)."""
    base_path = "processed_weather"
    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "test.csv"))
    # Encode target column "RainTomorrow" to binary (0 for No, 1 for Yes)
    y_train = train_df["RainTomorrow"].map({"No": 0, "Yes": 1})
    X_train = train_df.drop(columns=["RainTomorrow"])
    y_test = test_df["RainTomorrow"].map({"No": 0, "Yes": 1})
    X_test = test_df.drop(columns=["RainTomorrow"])
    return X_train, y_train, X_test, y_test


def preprocess(X_train, X_test):
    """Standardise features to zero mean and unit variance.

    Returns the transformed training and test sets.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_model(X_train, y_train, C=1.0, solver="liblinear"):
    """Instantiate and fit a LogisticRegression model.

    Parameters
    ----------
    X_train: array‑like, shape (n_samples, n_features)
        Training features.
    y_train: array‑like, shape (n_samples,)
        Training labels.
    C: float, default=1.0
        Inverse of regularisation strength; smaller values mean stronger regularisation.
    solver: str, default='liblinear'
        Solver suitable for small datasets.
    """
    model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    """Print evaluation metrics and return ROC data for plotting."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["No", "Yes"])

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}\n")
    print("Classification Report:\n", report)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return fpr, tpr, auc


def plot_roc(fpr, tpr, auc):
    """Plot the ROC curve and save it as 'roc_curve.png'."""
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()


def main():
    X_train, y_train, X_test, y_test = load_data()
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    fpr, tpr, auc = evaluate(model, X_test_scaled, y_test)
    plot_roc(fpr, tpr, auc)


if __name__ == "__main__":
    main()
