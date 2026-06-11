from __future__ import annotations

from typing import Any

import pandas as pd


def predict_regression_value(artifact: dict[str, Any], feature_values: dict[str, float]) -> float:
    """Predict a single scalar property from an aligned feature dictionary."""
    feature_names = artifact["feature_names"]
    input_frame = pd.DataFrame([{feature_name: float(feature_values.get(feature_name, 0.0)) for feature_name in feature_names}])
    prediction = artifact["model"].predict(input_frame)
    return float(prediction[0])


def predict_classification_value(artifact: dict[str, Any], feature_values: dict[str, float]) -> dict[str, Any]:
    """Predict a single class label and probability distribution."""
    feature_names = artifact["feature_names"]
    input_frame = pd.DataFrame([{feature_name: float(feature_values.get(feature_name, 0.0)) for feature_name in feature_names}])
    model = artifact["model"]
    label = model.predict(input_frame)[0]
    probabilities: dict[str, float]
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(input_frame)[0]
        classes = getattr(model, "classes_", artifact.get("classes") or [])
        probabilities = {
            str(class_label): float(probability)
            for class_label, probability in zip(classes, raw_probabilities, strict=False)
        }
    else:
        probabilities = {str(label): 1.0}
    return {"label": int(label) if isinstance(label, (int, float)) and float(label).is_integer() else label, "probabilities": probabilities}
