from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from chemstudio.ml import explainer


def test_shap_available_flag_returns_bool() -> None:
    assert isinstance(explainer.is_shap_available(), bool)


def test_global_importance_sorted() -> None:
    shap_values = np.asarray(
        [
            [1.0, 0.1, 0.4],
            [-3.0, 0.2, 0.6],
        ]
    )

    importance = explainer._global_importance(shap_values, ["a", "b", "c"])

    assert list(importance) == ["a", "c", "b"]
    assert importance["a"] == pytest.approx(2.0)


def test_explainer_raises_when_shap_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(explainer, "shap", None)

    with pytest.raises(explainer.SHAPUnavailableError, match="shap"):
        explainer.create_explainer(
            Pipeline([("imputer", SimpleImputer()), ("model", RandomForestRegressor())]),
            pd.DataFrame({"x": [1.0, 2.0]}),
            "random_forest",
        )


@pytest.mark.skipif(not explainer.is_shap_available(), reason="SHAP is not installed")
def test_tree_explainer_returns_shap_values() -> None:
    dataset = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            "target": [0.1, 1.0, 1.8, 3.2, 4.1, 5.2],
        }
    )
    x_frame = dataset[["f1", "f2"]]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=20, random_state=42)),
        ]
    )
    model.fit(x_frame, dataset["target"])
    artifact = {
        "model": model,
        "model_key": "random_forest",
        "problem_type": "regression",
        "feature_names": ["f1", "f2"],
    }

    explanation = explainer.explain_model(artifact, x_frame, x_frame.head(3), max_display=2)

    assert explanation.shap_values.shape == (3, 2)
    assert list(explanation.global_importance) == sorted(
        explanation.global_importance,
        key=explanation.global_importance.get,
        reverse=True,
    )


@pytest.mark.skipif(not explainer.is_shap_available(), reason="SHAP is not installed")
def test_explain_single_prediction() -> None:
    x_frame = pd.DataFrame({"f1": [0.0, 1.0, 2.0, 3.0], "f2": [3.0, 2.0, 1.0, 0.0]})
    y_values = pd.Series([0.0, 1.0, 2.0, 3.0])
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=20, random_state=42)),
        ]
    )
    model.fit(x_frame, y_values)
    artifact = {
        "model": model,
        "model_key": "random_forest",
        "problem_type": "regression",
        "feature_names": ["f1", "f2"],
    }
    explanation = explainer.explain_model(artifact, x_frame, x_frame.head(2), max_display=2)

    payload = explainer.explain_single_prediction(artifact, explanation, x_frame.head(1))

    assert payload["features"] == ["f1", "f2"]
    assert len(payload["shap_values"]) == 2
    assert "html" in payload
