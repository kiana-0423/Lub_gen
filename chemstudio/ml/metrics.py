from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute core regression metrics used by the UI."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(np.mean(np.square(y_true - y_pred))))
    r2 = float(r2_score(y_true, y_pred))
    return {"r2": r2, "mae": mae, "rmse": rmse}
