from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score, r2_score, recall_score


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute core regression metrics used by the UI."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(np.mean(np.square(y_true - y_pred))))
    r2 = float(r2_score(y_true, y_pred))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    """Compute core classification metrics used by the UI."""
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
