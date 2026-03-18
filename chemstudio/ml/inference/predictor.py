from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd


class Predictor:
    def predict(self, bundle: dict[str, object], feature_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        if not feature_rows:
            return []

        estimator = bundle["model"]
        feature_columns = list(bundle["feature_columns"])
        frame = pd.DataFrame(feature_rows, columns=feature_columns)
        raw_predictions = estimator.predict(frame)
        confidences = self._confidence_scores(estimator, frame, raw_predictions)
        problem_type = str(bundle["problem_type"])

        results: list[dict[str, object]] = []
        for row, prediction, confidence in zip(feature_rows, raw_predictions, confidences, strict=True):
            if problem_type == "classification":
                results.append(
                    {
                        "predicted_label": str(prediction),
                        "predicted_value": None,
                        "confidence": confidence,
                        "input_features": dict(row),
                    }
                )
            else:
                results.append(
                    {
                        "predicted_label": "",
                        "predicted_value": float(prediction),
                        "confidence": confidence,
                        "input_features": dict(row),
                    }
                )
        return results

    def _confidence_scores(self, estimator, frame: pd.DataFrame, predictions) -> list[float | None]:
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(frame)
            return [float(row.max()) for row in proba]

        estimators = getattr(estimator, "estimators_", None)
        if estimators:
            values = frame.to_numpy()
            per_estimator = pd.DataFrame([tree.predict(values) for tree in estimators])
            scores: list[float | None] = []
            for column_index, predicted in enumerate(predictions):
                std = float(per_estimator.iloc[:, column_index].std())
                scale = abs(float(predicted)) + 1.0
                scores.append(float(1.0 / (1.0 + (std / scale))))
            return scores

        return [None for _ in range(len(frame.index))]
