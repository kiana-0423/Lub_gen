from __future__ import annotations

from typing import Any

import pandas as pd


def predict_regression_value(artifact: dict[str, Any], feature_values: dict[str, float]) -> float:
    """Predict a single scalar property from an aligned feature dictionary."""
    feature_names = artifact["feature_names"]
    input_frame = pd.DataFrame([{feature_name: float(feature_values.get(feature_name, 0.0)) for feature_name in feature_names}])
    prediction = artifact["model"].predict(input_frame)
    return float(prediction[0])
