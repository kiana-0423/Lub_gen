from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from chemstudio.ml.base import ProblemType

_XGBOOST_IMPORT_ERROR: Exception | None = None

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    XGBRegressor = None
    _XGBOOST_IMPORT_ERROR = exc


MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "key": "random_forest",
        "label": "RandomForestRegressor",
        "type": "regression",
        "available": True,
    },
    {"key": "xgboost", "label": "XGBoost", "type": "regression", "available": XGBRegressor is not None},
    {"key": "svr", "label": "SVR", "type": "regression", "available": True},
    {
        "key": "linear_regression",
        "label": "LinearRegression",
        "type": "regression",
        "available": True,
    },
    {
        "key": "random_forest_classifier",
        "label": "RandomForestClassifier",
        "type": "classification",
        "available": True,
    },
    {
        "key": "gradient_boosting_classifier",
        "label": "GradientBoostingClassifier",
        "type": "classification",
        "available": True,
    },
    {"key": "svc", "label": "SVC", "type": "classification", "available": True},
    {
        "key": "logistic_regression",
        "label": "LogisticRegression",
        "type": "classification",
        "available": True,
    },
]


def get_model_catalog(problem_type: ProblemType | None = None) -> list[dict[str, Any]]:
    """Return supported model metadata, optionally filtered by problem type."""
    items = MODEL_CATALOG
    if problem_type is not None:
        items = [item for item in items if item["type"] == problem_type]
    return [item.copy() for item in items]


def get_model_row(model_key: str) -> dict[str, Any]:
    model_row = next((item for item in MODEL_CATALOG if item["key"] == model_key), None)
    if model_row is None:
        raise ValueError(f"Unknown model key: {model_key}")
    if not bool(model_row["available"]):
        raise RuntimeError(f"Model `{model_row['label']}` is not available in this environment.")
    return model_row.copy()


def create_regression_model(model_key: str) -> Pipeline:
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
            message = "XGBoost is not available. Install `xgboost` to enable this model."
            if _XGBOOST_IMPORT_ERROR is not None:
                message = (
                    "XGBoost is installed but could not be loaded. On macOS, install the OpenMP "
                    f"runtime with `brew install libomp`. Details: {_XGBOOST_IMPORT_ERROR}"
                )
            raise RuntimeError(message)
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

    raise ValueError(f"Unsupported regression model key: {model_key}")


def create_classification_model(model_key: str) -> Pipeline:
    """Build a classification model pipeline from the selected key."""
    if model_key == "random_forest_classifier":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
            ]
        )

    if model_key == "gradient_boosting_classifier":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        )

    if model_key == "svc":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=10.0, probability=True, random_state=42)),
            ]
        )

    if model_key == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

    raise ValueError(f"Unsupported classification model key: {model_key}")


def create_model(model_key: str, problem_type: ProblemType | None = None) -> Pipeline:
    """Build a model pipeline, inferring type from the catalog when omitted."""
    model_row = get_model_row(model_key)
    effective_type = problem_type or str(model_row["type"])
    if effective_type == "classification":
        return create_classification_model(model_key)
    if effective_type == "regression":
        return create_regression_model(model_key)
    raise ValueError(f"Unsupported problem type: {effective_type}")
