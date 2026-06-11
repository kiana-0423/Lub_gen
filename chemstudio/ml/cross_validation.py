from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from chemstudio.ml.base import ProblemType
from chemstudio.ml.metrics import calculate_classification_metrics, calculate_regression_metrics


def _effective_fold_count(y_values: pd.Series, problem_type: ProblemType, n_folds: int) -> int:
    requested = max(2, int(n_folds))
    if problem_type == "classification":
        class_counts = y_values.value_counts()
        if class_counts.empty:
            return 2
        min_class_count = int(class_counts.min())
        if min_class_count < 2:
            raise ValueError("Classification cross-validation requires at least 2 samples per class.")
        return min(requested, min_class_count, int(len(y_values)))
    return max(2, min(requested, int(len(y_values))))


def cross_validate(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: ProblemType,
    n_folds: int = 5,
    scoring: str | None = None,
) -> dict[str, Any]:
    """Run K-fold CV and return aggregate score plus fold-level metrics."""
    fold_count = _effective_fold_count(y, problem_type, n_folds)
    if problem_type == "classification":
        splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42)
        splits = splitter.split(X, y)
        score_key = scoring or "f1"
    else:
        splitter = KFold(n_splits=fold_count, shuffle=True, random_state=42)
        splits = splitter.split(X)
        score_key = scoring or "r2"

    cv_scores: list[float] = []
    cv_details: list[dict[str, Any]] = []
    for fold_index, (train_index, test_index) in enumerate(splits, start=1):
        fold_pipeline = clone(pipeline)
        x_train = X.iloc[train_index]
        x_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        fold_pipeline.fit(x_train, y_train)
        y_pred = fold_pipeline.predict(x_test)
        if problem_type == "classification":
            metrics = calculate_classification_metrics(np.asarray(y_test), np.asarray(y_pred))
        else:
            metrics = calculate_regression_metrics(
                np.asarray(y_test, dtype=float),
                np.asarray(y_pred, dtype=float),
            )
        score = float(metrics.get(score_key, metrics.get("r2", metrics.get("accuracy", 0.0))))
        cv_scores.append(score)
        cv_details.append({"fold": fold_index, "metrics": metrics})

    return {
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "cv_scores": cv_scores,
        "cv_details": cv_details,
        "n_folds": int(fold_count),
        "scoring": score_key,
    }
