from __future__ import annotations

from dataclasses import dataclass, field
import html
import tempfile
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

from chemstudio.ml.base import ProblemType

try:  # pragma: no cover - optional dependency
    import shap
except ImportError:  # pragma: no cover
    shap = None

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None


ExplainerKind = Literal["tree", "linear", "kernel"]


class SHAPUnavailableError(RuntimeError):
    """Raised when SHAP functionality is requested without the optional dependency."""


@dataclass
class ExplainerBundle:
    """Wrap a SHAP explainer with the preprocessing context needed by a pipeline."""

    explainer: Any
    kind: ExplainerKind
    feature_names: list[str]
    model: Any
    preprocessor: Any | None = None


@dataclass
class SHAPExplanation:
    """SHAP explanation payload used by services and UI widgets."""

    shap_values: np.ndarray
    feature_names: list[str]
    base_value: float
    global_importance: dict[str, float]
    summary_plot_path: str
    problem_type: ProblemType
    explainer: ExplainerBundle | None = field(default=None, repr=False)


def is_shap_available() -> bool:
    """Return whether the optional SHAP dependency is importable."""
    return shap is not None


def create_explainer(
    model: Any,
    x_train: pd.DataFrame,
    model_key: str,
    *,
    problem_type: ProblemType | None = None,
) -> ExplainerBundle:
    """Create a SHAP explainer for the fitted model or pipeline."""
    _require_shap()
    training_frame = _as_numeric_frame(x_train)
    final_model, preprocessor = _split_pipeline(model)
    transformed_train = _transform_frame(preprocessor, training_frame)

    if _is_tree_model(final_model, model_key):
        return ExplainerBundle(
            explainer=shap.TreeExplainer(final_model),
            kind="tree",
            feature_names=list(training_frame.columns),
            model=model,
            preprocessor=preprocessor,
        )

    if _is_linear_model(final_model, model_key):
        return ExplainerBundle(
            explainer=shap.LinearExplainer(final_model, _sample_frame(transformed_train, 100)),
            kind="linear",
            feature_names=list(training_frame.columns),
            model=model,
            preprocessor=preprocessor,
        )

    background = _sample_frame(training_frame, 100)
    predict_fn = _make_prediction_function(model, list(training_frame.columns), problem_type)
    return ExplainerBundle(
        explainer=shap.KernelExplainer(predict_fn, background),
        kind="kernel",
        feature_names=list(training_frame.columns),
        model=model,
        preprocessor=None,
    )


def explain_model(
    artifact: dict[str, Any],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    max_display: int = 20,
) -> SHAPExplanation:
    """Compute SHAP values, global importance, and a summary plot for a model artifact."""
    _require_shap()
    feature_names = [str(feature) for feature in artifact["feature_names"]]
    problem_type = str(artifact.get("problem_type", "regression"))
    if problem_type not in {"regression", "classification"}:
        raise ValueError(f"Unsupported problem type for SHAP: {problem_type}")

    train_frame = _align_frame(x_train, feature_names)
    test_frame = _align_frame(x_test, feature_names).head(200)
    if test_frame.empty:
        raise ValueError("SHAP explanation requires at least one sample.")

    bundle = create_explainer(
        artifact["model"],
        train_frame,
        str(artifact.get("model_key", "")),
        problem_type=problem_type,  # type: ignore[arg-type]
    )
    shap_values = _compute_shap_values(bundle, test_frame, problem_type)  # type: ignore[arg-type]
    base_value = _extract_base_value(bundle.explainer, problem_type)  # type: ignore[arg-type]
    global_importance = _global_importance(shap_values, feature_names)
    summary_plot_path = _save_summary_plot(shap_values, test_frame, max_display=max_display)

    return SHAPExplanation(
        shap_values=shap_values,
        feature_names=feature_names,
        base_value=base_value,
        global_importance=global_importance,
        summary_plot_path=str(summary_plot_path),
        problem_type=problem_type,  # type: ignore[arg-type]
        explainer=bundle,
    )


def explain_single_prediction(
    artifact: dict[str, Any],
    explanation: SHAPExplanation | ExplainerBundle,
    feature_values: pd.DataFrame,
) -> dict[str, Any]:
    """Return local SHAP contribution data and compact HTML for one prediction."""
    _require_shap()
    feature_names = [str(feature) for feature in artifact["feature_names"]]
    problem_type = str(artifact.get("problem_type", "regression"))
    bundle = explanation.explainer if isinstance(explanation, SHAPExplanation) else explanation
    if bundle is None:
        raise ValueError("The SHAP explanation does not include an explainer cache.")

    sample_frame = _align_frame(feature_values, feature_names).head(1)
    shap_row = _compute_shap_values(bundle, sample_frame, problem_type)[0]  # type: ignore[arg-type]
    feature_row = sample_frame.iloc[0].to_dict()
    prediction = artifact["model"].predict(sample_frame)
    prediction_value = _extract_prediction_value(prediction, problem_type)  # type: ignore[arg-type]
    base_value = _extract_base_value(bundle.explainer, problem_type)  # type: ignore[arg-type]
    contribution_rows = sorted(
        (
            {
                "feature": feature,
                "value": float(feature_row[feature]),
                "shap_value": float(shap_row[index]),
            }
            for index, feature in enumerate(feature_names)
        ),
        key=lambda row: abs(row["shap_value"]),
        reverse=True,
    )

    return {
        "base_value": base_value,
        "prediction": prediction_value,
        "features": feature_names,
        "feature_values": {key: float(value) for key, value in feature_row.items()},
        "shap_values": [float(value) for value in shap_row],
        "top_contributions": contribution_rows[:20],
        "html": _build_force_html(base_value, prediction_value, contribution_rows[:20]),
    }


def _require_shap() -> None:
    if shap is None:
        raise SHAPUnavailableError("请安装 shap 库以启用模型解释功能。")


def _split_pipeline(model: Any) -> tuple[Any, Any | None]:
    if isinstance(model, Pipeline):
        if len(model.steps) == 0:
            raise ValueError("Pipeline model contains no steps.")
        final_model = model.steps[-1][1]
        preprocessor = model[:-1] if len(model.steps) > 1 else None
        return final_model, preprocessor
    return model, None


def _as_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)


def _align_frame(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    aligned = frame.copy()
    for feature_name in feature_names:
        if feature_name not in aligned.columns:
            aligned[feature_name] = 0.0
    return _as_numeric_frame(aligned[feature_names])


def _sample_frame(frame: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    if len(frame) <= sample_count:
        return frame
    if shap is not None:
        sampled = shap.sample(frame, sample_count, random_state=42)
        if isinstance(sampled, pd.DataFrame):
            return sampled
    return frame.sample(n=sample_count, random_state=42)


def _transform_frame(preprocessor: Any | None, frame: pd.DataFrame) -> pd.DataFrame:
    if preprocessor is None:
        return frame
    transformed = preprocessor.transform(frame)
    return pd.DataFrame(np.asarray(transformed), columns=list(frame.columns), index=frame.index)


def _is_tree_model(model: Any, model_key: str) -> bool:
    tree_types: tuple[type[Any], ...] = (
        RandomForestRegressor,
        RandomForestClassifier,
        GradientBoostingRegressor,
        GradientBoostingClassifier,
    )
    optional_tree_types = tuple(
        candidate for candidate in (XGBRegressor, XGBClassifier) if candidate is not None
    )
    return isinstance(model, tree_types + optional_tree_types) or model_key in {
        "random_forest",
        "random_forest_classifier",
        "gradient_boosting_classifier",
        "xgboost",
    }


def _is_linear_model(model: Any, model_key: str) -> bool:
    return isinstance(model, (LinearRegression, LogisticRegression)) or model_key in {
        "linear_regression",
        "logistic_regression",
    }


def _make_prediction_function(
    model: Any,
    feature_names: list[str],
    problem_type: ProblemType | None,
):
    def predict(values: np.ndarray | pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(values, columns=feature_names)
        if problem_type == "classification" and hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(frame), dtype=float)
            if probabilities.ndim == 2 and probabilities.shape[1] > 1:
                return probabilities[:, 1]
            return probabilities.ravel()
        return np.asarray(model.predict(frame), dtype=float)

    return predict


def _compute_shap_values(bundle: ExplainerBundle, frame: pd.DataFrame, problem_type: ProblemType) -> np.ndarray:
    explain_frame = _transform_frame(bundle.preprocessor, frame) if bundle.kind in {"tree", "linear"} else frame
    if bundle.kind == "kernel":
        raw_values = bundle.explainer.shap_values(explain_frame, nsamples="auto")
    else:
        raw_values = bundle.explainer.shap_values(explain_frame)
    return _normalize_shap_values(raw_values, len(explain_frame), problem_type)


def _normalize_shap_values(raw_values: Any, sample_count: int, problem_type: ProblemType) -> np.ndarray:
    if hasattr(raw_values, "values"):
        raw_values = raw_values.values
    if isinstance(raw_values, list):
        raw_values = raw_values[1] if problem_type == "classification" and len(raw_values) > 1 else raw_values[0]

    values = np.asarray(raw_values, dtype=float)
    if values.ndim == 3:
        if values.shape[0] == sample_count:
            values = values[:, :, 1] if problem_type == "classification" and values.shape[2] > 1 else values[:, :, 0]
        else:
            values = values[1] if problem_type == "classification" and values.shape[0] > 1 else values[0]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def _extract_base_value(explainer: Any, problem_type: ProblemType) -> float:
    expected_value = getattr(explainer, "expected_value", 0.0)
    if isinstance(expected_value, list):
        expected_value = expected_value[1] if problem_type == "classification" and len(expected_value) > 1 else expected_value[0]
    values = np.asarray(expected_value, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    if problem_type == "classification" and values.size > 1:
        return float(values[1])
    return float(values[0])


def _global_importance(shap_values: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    importance = np.mean(np.abs(shap_values), axis=0)
    rows = sorted(
        ((feature, float(score)) for feature, score in zip(feature_names, importance, strict=False)),
        key=lambda row: row[1],
        reverse=True,
    )
    return dict(rows)


def _save_summary_plot(shap_values: np.ndarray, x_frame: pd.DataFrame, *, max_display: int) -> Path:
    output = Path(tempfile.NamedTemporaryFile(prefix="chemstudio_shap_", suffix=".png", delete=False).name)
    plt.figure(figsize=(8, 4.8), dpi=100)
    try:
        shap.summary_plot(shap_values, x_frame, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output, bbox_inches="tight", dpi=100)
    finally:
        plt.close()
    return output


def _extract_prediction_value(prediction: Any, problem_type: ProblemType) -> float:
    values = np.asarray(prediction, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    return float(values[0])


def _build_force_html(base_value: float, prediction: float, rows: list[dict[str, float | str]]) -> str:
    max_abs = max((abs(float(row["shap_value"])) for row in rows), default=1.0)
    body_rows = []
    for row in rows:
        shap_value = float(row["shap_value"])
        width = 8.0 + 82.0 * abs(shap_value) / max_abs if max_abs else 8.0
        color = "#B71C1C" if shap_value >= 0 else "#0B3D91"
        sign = "+" if shap_value >= 0 else ""
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['feature']))}</td>"
            f"<td>{float(row['value']):.4g}</td>"
            f"<td style='color:{color}; font-weight:600'>{sign}{shap_value:.4g}</td>"
            f"<td><div style='background:{color}; height:10px; width:{width:.1f}%;'></div></td>"
            "</tr>"
        )
    return (
        "<html><body>"
        "<h3>单样本 SHAP 局部解释</h3>"
        f"<p>基线值: <b>{base_value:.4g}</b> | 预测值: <b>{prediction:.4g}</b></p>"
        "<table border='0' cellspacing='4' cellpadding='3'>"
        "<tr><th align='left'>特征</th><th align='right'>取值</th><th align='right'>贡献</th><th align='left'>方向/强度</th></tr>"
        + "".join(body_rows)
        + "</table></body></html>"
    )
