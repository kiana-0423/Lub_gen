from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from chemstudio.ml.metrics import calculate_regression_metrics

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None


MODEL_CATALOG: list[dict[str, Any]] = [
    {"key": "random_forest", "label": "RandomForestRegressor", "available": True},
    {"key": "xgboost", "label": "XGBoost", "available": XGBRegressor is not None},
    {"key": "svr", "label": "SVR", "available": True},
    {"key": "linear_regression", "label": "LinearRegression", "available": True},
]


def get_model_catalog() -> list[dict[str, Any]]:
    """Return supported model metadata for the GUI."""
    return [item.copy() for item in MODEL_CATALOG]


def create_model(model_key: str) -> Pipeline:
    """Build a regression model pipeline from the selected key."""
    if model_key == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
            ]
        )

    if model_key == "svr":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
            ]
        )

    if model_key == "linear_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

    if model_key == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("XGBoost is not installed. Install `xgboost` to enable this model.")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=42,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unsupported model key: {model_key}")


def train_regression_model(
    dataset: pd.DataFrame,
    target_name: str,
    feature_names: list[str],
    model_key: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a regression model and return the fitted artifact with metrics."""
    if target_name not in dataset.columns:
        raise ValueError(f"Target column `{target_name}` was not found in the dataset.")
    if not feature_names:
        raise ValueError("No usable feature columns were detected.")

    model_row = next((item for item in MODEL_CATALOG if item["key"] == model_key), None)
    if model_row is None:
        raise ValueError(f"Unknown model key: {model_key}")

    training_frame = dataset[feature_names + [target_name]].copy()
    training_frame = training_frame.dropna(subset=[target_name])
    if len(training_frame) < 4:
        raise ValueError("Training requires at least 4 rows with a non-empty target value.")

    x_frame = training_frame[feature_names]
    y_values = training_frame[target_name].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x_frame,
        y_values,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = create_model(model_key)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    y_true_array = np.asarray(y_test, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    metrics = calculate_regression_metrics(y_true_array, y_pred_array)

    return {
        "model": pipeline,
        "model_key": model_key,
        "model_name": str(model_row["label"]),
        "target_name": target_name,
        "feature_names": feature_names,
        "metrics": metrics,
        "sample_count": int(len(training_frame)),
        "test_size": float(test_size),
        "y_true": y_true_array.tolist(),
        "y_pred": y_pred_array.tolist(),
    }
