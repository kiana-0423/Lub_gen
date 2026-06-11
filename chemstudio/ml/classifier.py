from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from chemstudio.ml.cross_validation import cross_validate
from chemstudio.ml.hyperparameter_search import search_hyperparameters
from chemstudio.ml.metrics import calculate_classification_metrics
from chemstudio.ml.model_catalog import create_classification_model, get_model_row


def create_model(model_key: str):
    """Build a classification model pipeline from the selected key."""
    return create_classification_model(model_key)


def train_classification_model(
    dataset: pd.DataFrame,
    target_name: str,
    feature_names: list[str],
    model_key: str,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_mode: bool = False,
    n_folds: int = 5,
    hp_search: bool = False,
    hp_method: str = "grid",
    hp_n_iter: int = 20,
) -> dict[str, Any]:
    """Train a classification model and return the fitted artifact with metrics."""
    if target_name not in dataset.columns:
        raise ValueError(f"Target column `{target_name}` was not found in the dataset.")
    if not feature_names:
        raise ValueError("No usable feature columns were detected.")

    model_row = get_model_row(model_key)
    if model_row["type"] != "classification":
        raise ValueError(f"Model `{model_key}` is not a classification model.")

    training_frame = dataset[feature_names + [target_name]].copy()
    training_frame = training_frame.dropna(subset=[target_name])
    if len(training_frame) < 4:
        raise ValueError("Training requires at least 4 rows with a non-empty target value.")

    x_frame = training_frame[feature_names]
    y_values = training_frame[target_name].astype(int)
    class_counts = y_values.value_counts()
    stratify = y_values if int(class_counts.min()) >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x_frame,
        y_values,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = create_model(model_key)
    cv_results = None
    hp_results = None
    if cv_mode:
        cv_results = cross_validate(pipeline, x_frame, y_values, "classification", n_folds=n_folds)
    if hp_search:
        hp_payload = search_hyperparameters(
            pipeline,
            x_frame,
            y_values,
            model_key=model_key,
            problem_type="classification",
            method=hp_method,
            n_iter=hp_n_iter,
            cv_folds=n_folds,
        )
        pipeline = hp_payload.pop("best_estimator")
        hp_results = hp_payload

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_true_array = np.asarray(y_test)
    y_pred_array = np.asarray(y_pred)
    metrics = calculate_classification_metrics(y_true_array, y_pred_array)

    return {
        "model": pipeline,
        "model_key": model_key,
        "model_name": str(model_row["label"]),
        "problem_type": "classification",
        "target_name": target_name,
        "feature_names": feature_names,
        "metrics": metrics,
        "sample_count": int(len(training_frame)),
        "test_size": float(test_size),
        "classes": [int(label) for label in getattr(pipeline, "classes_", metrics["labels"])],
        "y_true": y_true_array.astype(int).tolist(),
        "y_pred": y_pred_array.astype(int).tolist(),
        "cv_results": cv_results,
        "hp_results": hp_results,
    }
