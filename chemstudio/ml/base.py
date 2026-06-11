from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ProblemType = Literal["regression", "classification"]


@dataclass
class ModelArtifact:
    """Common trained-model artifact shape used by services and persistence."""

    model: Any
    model_key: str
    model_name: str
    problem_type: ProblemType
    target_name: str
    feature_names: list[str]
    metrics: dict[str, Any]
    sample_count: int
    test_size: float
    y_true: list[Any]
    y_pred: list[Any]
    cv_results: dict[str, Any] | None = None
    hp_results: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "model_key": self.model_key,
            "model_name": self.model_name,
            "problem_type": self.problem_type,
            "target_name": self.target_name,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "sample_count": int(self.sample_count),
            "test_size": float(self.test_size),
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "cv_results": self.cv_results,
            "hp_results": self.hp_results,
        }
        payload.update(self.extra)
        return payload
