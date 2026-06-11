from __future__ import annotations

import time
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold

from chemstudio.ml.base import ProblemType


SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 6, 12],
        "model__min_samples_split": [2, 5],
    },
    "svr": {
        "model__C": [1.0, 10.0, 50.0],
        "model__epsilon": [0.05, 0.1, 0.2],
        "model__gamma": ["scale", "auto"],
    },
    "linear_regression": {},
    "xgboost": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.03, 0.05, 0.1],
    },
    "random_forest_classifier": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 6, 12],
        "model__min_samples_split": [2, 5],
    },
    "gradient_boosting_classifier": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.03, 0.1],
        "model__max_depth": [2, 3],
    },
    "svc": {
        "model__C": [1.0, 10.0, 50.0],
        "model__gamma": ["scale", "auto"],
    },
    "logistic_regression": {
        "model__C": [0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
    },
}


def _make_cv(y: pd.Series, problem_type: ProblemType, cv_folds: int):
    fold_count = max(2, min(int(cv_folds), int(len(y))))
    if problem_type == "classification":
        min_class_count = int(y.value_counts().min())
        if min_class_count < 2:
            raise ValueError("Classification hyperparameter search requires at least 2 samples per class.")
        fold_count = min(fold_count, min_class_count)
        return StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42)
    return KFold(n_splits=fold_count, shuffle=True, random_state=42)


def search_hyperparameters(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_key: str,
    problem_type: ProblemType,
    method: str = "grid",
    n_iter: int = 20,
    cv_folds: int = 5,
    scoring: str | None = None,
) -> dict[str, Any]:
    """Search hyperparameters and return the best estimator with compact results."""
    search_space = SEARCH_SPACES.get(model_key, {})
    if not search_space:
        fitted = clone(pipeline)
        fitted.fit(X, y)
        return {
            "best_estimator": fitted,
            "best_params": {},
            "best_score": None,
            "search_results": [],
            "search_time": 0.0,
            "method": method,
            "cv_folds": 0,
        }

    scoring_name = scoring or ("f1_weighted" if problem_type == "classification" else "r2")
    cv = _make_cv(y, problem_type, cv_folds)
    start_time = time.perf_counter()
    if method == "random":
        search = RandomizedSearchCV(
            estimator=clone(pipeline),
            param_distributions=search_space,
            n_iter=max(1, int(n_iter)),
            cv=cv,
            scoring=scoring_name,
            random_state=42,
            n_jobs=None,
        )
    else:
        search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=search_space,
            cv=cv,
            scoring=scoring_name,
            n_jobs=None,
        )

    search.fit(X, y)
    elapsed = float(time.perf_counter() - start_time)
    params = search.cv_results_["params"]
    mean_scores = search.cv_results_["mean_test_score"]
    std_scores = search.cv_results_["std_test_score"]
    ranked = sorted(
        (
            {
                "params": dict(param_set),
                "mean_test_score": float(mean_score),
                "std_test_score": float(std_score),
            }
            for param_set, mean_score, std_score in zip(params, mean_scores, std_scores, strict=False)
        ),
        key=lambda item: item["mean_test_score"],
        reverse=True,
    )
    return {
        "best_estimator": search.best_estimator_,
        "best_params": dict(search.best_params_),
        "best_score": float(search.best_score_),
        "search_results": ranked[:20],
        "search_time": elapsed,
        "method": method,
        "cv_folds": int(getattr(cv, "n_splits", cv_folds)),
        "scoring": scoring_name,
    }
