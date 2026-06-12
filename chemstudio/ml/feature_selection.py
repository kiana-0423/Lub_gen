from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold

from chemstudio.ml.base import ProblemType


logger = logging.getLogger(__name__)

FeatureSelectionStrategy = Literal["none", "variance", "correlation", "univariate", "model_based", "full"]

DEFAULT_PROTECTED_FEATURES = (
    "mol_wt",
    "mol_logp",
    "tpsa",
    "h_donors",
    "h_acceptors",
    "rotatable_bonds",
    "ring_count",
    "fraction_csp3",
)


@dataclass
class FeatureSelectionStage:
    """One step in a feature-selection run."""

    name: str
    before_count: int
    after_count: int
    removed_features: list[str] = field(default_factory=list)
    from_cache: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "before_count": int(self.before_count),
            "after_count": int(self.after_count),
            "removed_count": int(len(self.removed_features)),
            "removed_features": list(self.removed_features),
            "from_cache": bool(self.from_cache),
        }


@dataclass
class FeatureSelectionReport:
    """Traceable report for a feature-selection run."""

    strategy: FeatureSelectionStrategy
    problem_type: ProblemType
    initial_feature_count: int
    final_feature_count: int
    selected_features: list[str]
    protected_features: list[str]
    stages: list[FeatureSelectionStage] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "problem_type": self.problem_type,
            "initial_feature_count": int(self.initial_feature_count),
            "final_feature_count": int(self.final_feature_count),
            "selected_features": list(self.selected_features),
            "protected_features": list(self.protected_features),
            "stages": [stage.to_dict() for stage in self.stages],
        }


class FeatureSelector:
    """Run staged feature selection for Mordred-heavy molecular feature matrices."""

    def __init__(
        self,
        *,
        strategy: FeatureSelectionStrategy = "full",
        max_features: int = 150,
        protected_features: tuple[str, ...] = DEFAULT_PROTECTED_FEATURES,
        variance_threshold: float = 0.01,
        missing_threshold: float = 0.30,
        correlation_threshold: float = 0.95,
        random_state: int = 42,
    ) -> None:
        if strategy not in {"none", "variance", "correlation", "univariate", "model_based", "full"}:
            raise ValueError(f"Unsupported feature-selection strategy: {strategy}")
        self.strategy = strategy
        self.max_features = max(1, int(max_features))
        self.protected_features = protected_features
        self.variance_threshold = float(variance_threshold)
        self.missing_threshold = float(missing_threshold)
        self.correlation_threshold = float(correlation_threshold)
        self.random_state = int(random_state)
        self._cache: dict[str, list[str]] = {}

    def clear_cache(self) -> None:
        """Clear cached intermediate feature-selection stages."""
        self._cache.clear()

    def select(
        self,
        x_frame: pd.DataFrame,
        y_values: pd.Series,
        problem_type: ProblemType,
    ) -> tuple[list[str], FeatureSelectionReport]:
        """Return selected feature names plus a traceable report."""
        if problem_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported problem type: {problem_type}")
        if x_frame.empty:
            raise ValueError("Feature selection requires at least one feature column.")

        numeric_frame = x_frame.apply(pd.to_numeric, errors="coerce")
        selected_features = list(numeric_frame.columns)
        protected = [feature for feature in self.protected_features if feature in selected_features]
        stages: list[FeatureSelectionStage] = []

        if self.strategy == "none":
            report = FeatureSelectionReport(
                strategy=self.strategy,
                problem_type=problem_type,
                initial_feature_count=len(selected_features),
                final_feature_count=len(selected_features),
                selected_features=selected_features,
                protected_features=protected,
            )
            return selected_features, report

        stage_names = self._stage_names()
        if "variance" in stage_names:
            selected_features = self._apply_variance_filter(numeric_frame, selected_features, protected, stages)
        if "missing" in stage_names:
            selected_features = self._apply_missing_filter(numeric_frame, selected_features, protected, stages)
        if "correlation" in stage_names:
            selected_features = self._apply_correlation_filter(numeric_frame, selected_features, protected, stages)
        if "univariate" in stage_names:
            selected_features = self._apply_univariate_filter(
                numeric_frame,
                y_values,
                selected_features,
                protected,
                problem_type,
                stages,
            )
        if "model_based" in stage_names:
            selected_features = self._apply_model_based_filter(
                numeric_frame,
                y_values,
                selected_features,
                protected,
                problem_type,
                stages,
            )

        selected_features = self._with_protected(selected_features, protected)
        report = FeatureSelectionReport(
            strategy=self.strategy,
            problem_type=problem_type,
            initial_feature_count=len(numeric_frame.columns),
            final_feature_count=len(selected_features),
            selected_features=selected_features,
            protected_features=protected,
            stages=stages,
        )
        return selected_features, report

    def _stage_names(self) -> list[str]:
        stages_by_strategy = {
            "variance": ["variance", "missing"],
            "correlation": ["variance", "missing", "correlation"],
            "univariate": ["variance", "missing", "correlation", "univariate"],
            "model_based": ["variance", "missing", "correlation", "model_based"],
            "full": ["variance", "missing", "correlation", "univariate", "model_based"],
        }
        return stages_by_strategy.get(self.strategy, [])

    def _apply_variance_filter(
        self,
        frame: pd.DataFrame,
        selected_features: list[str],
        protected: list[str],
        stages: list[FeatureSelectionStage],
    ) -> list[str]:
        before = list(selected_features)
        if not before:
            return before
        cache_key = self._cache_key("variance", frame[before])
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Feature selection stage '%s' served from cache", "variance")
            kept = self._with_protected(list(cached), protected)
            self._append_stage(stages, "variance", before, kept, from_cache=True)
            return kept

        selector = VarianceThreshold(threshold=self.variance_threshold)
        x_values = self._impute_frame(frame[before])
        try:
            selector.fit(x_values)
            kept = [feature for feature, keep in zip(before, selector.get_support(), strict=False) if keep]
        except ValueError:
            kept = []
        self._cache[cache_key] = list(kept)
        kept = self._with_protected(kept, protected)
        self._append_stage(stages, "variance", before, kept)
        return kept

    def _apply_missing_filter(
        self,
        frame: pd.DataFrame,
        selected_features: list[str],
        protected: list[str],
        stages: list[FeatureSelectionStage],
    ) -> list[str]:
        before = list(selected_features)
        cache_key = self._cache_key("missing", frame[before])
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Feature selection stage '%s' served from cache", "missing")
            kept = self._with_protected(list(cached), protected)
            self._append_stage(stages, "missing", before, kept, from_cache=True)
            return kept

        missing_rates = frame[before].isna().mean()
        kept = [feature for feature in before if float(missing_rates.get(feature, 0.0)) <= self.missing_threshold]
        self._cache[cache_key] = list(kept)
        kept = self._with_protected(kept, protected)
        self._append_stage(stages, "missing", before, kept)
        return kept

    def _apply_correlation_filter(
        self,
        frame: pd.DataFrame,
        selected_features: list[str],
        protected: list[str],
        stages: list[FeatureSelectionStage],
    ) -> list[str]:
        before = list(selected_features)
        if len(before) <= 1:
            self._append_stage(stages, "correlation", before, before)
            return before

        x_values = self._impute_frame(frame[before])
        correlation = x_values.corr().abs().fillna(0.0)
        to_drop: set[str] = set()
        ordered_features = list(correlation.columns)
        for left_index, left_feature in enumerate(ordered_features):
            if left_feature in to_drop:
                continue
            for right_feature in ordered_features[left_index + 1 :]:
                if right_feature in to_drop:
                    continue
                if float(correlation.loc[left_feature, right_feature]) <= self.correlation_threshold:
                    continue
                drop_candidate = self._choose_correlated_drop(left_feature, right_feature, protected)
                if drop_candidate is not None:
                    to_drop.add(drop_candidate)
        kept = [feature for feature in before if feature not in to_drop]
        kept = self._with_protected(kept, protected)
        self._append_stage(stages, "correlation", before, kept)
        return kept

    def _apply_univariate_filter(
        self,
        frame: pd.DataFrame,
        y_values: pd.Series,
        selected_features: list[str],
        protected: list[str],
        problem_type: ProblemType,
        stages: list[FeatureSelectionStage],
    ) -> list[str]:
        before = list(selected_features)
        if len(before) <= self.max_features:
            self._append_stage(stages, "univariate", before, before)
            return before
        x_values = self._impute_frame(frame[before])
        score_func = mutual_info_classif if problem_type == "classification" else mutual_info_regression
        selector = SelectKBest(
            score_func=lambda x, y: score_func(x, y, random_state=self.random_state),
            k=min(self.max_features, len(before)),
        )
        selector.fit(x_values, y_values)
        kept = [feature for feature, keep in zip(before, selector.get_support(), strict=False) if keep]
        kept = self._with_protected(kept, protected)
        self._append_stage(stages, "univariate", before, kept)
        return kept

    def _apply_model_based_filter(
        self,
        frame: pd.DataFrame,
        y_values: pd.Series,
        selected_features: list[str],
        protected: list[str],
        problem_type: ProblemType,
        stages: list[FeatureSelectionStage],
    ) -> list[str]:
        before = list(selected_features)
        if len(before) <= 1:
            self._append_stage(stages, "model_based", before, before)
            return before
        x_values = self._impute_frame(frame[before])
        estimator = (
            RandomForestClassifier(n_estimators=300, random_state=self.random_state)
            if problem_type == "classification"
            else RandomForestRegressor(n_estimators=300, random_state=self.random_state)
        )
        selector = SelectFromModel(
            estimator=estimator,
            threshold="median",
            max_features=min(self.max_features, len(before)),
        )
        selector.fit(x_values, y_values)
        kept = [feature for feature, keep in zip(before, selector.get_support(), strict=False) if keep]
        if not kept:
            importances = getattr(selector.estimator_, "feature_importances_", np.zeros(len(before)))
            ranked_indices = np.argsort(importances)[::-1]
            kept = [before[index] for index in ranked_indices[: min(self.max_features, len(before))]]
        kept = self._with_protected(kept, protected)
        self._append_stage(stages, "model_based", before, kept)
        return kept

    def _choose_correlated_drop(self, left_feature: str, right_feature: str, protected: list[str]) -> str | None:
        left_protected = left_feature in protected
        right_protected = right_feature in protected
        if left_protected and right_protected:
            return None
        if right_protected:
            return left_feature
        return right_feature

    def _impute_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
        medians = numeric_frame.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return numeric_frame.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)

    def _append_stage(
        self,
        stages: list[FeatureSelectionStage],
        name: str,
        before: list[str],
        after: list[str],
        *,
        from_cache: bool = False,
    ) -> None:
        after_set = set(after)
        stages.append(
            FeatureSelectionStage(
                name=name,
                before_count=len(before),
                after_count=len(after),
                removed_features=[feature for feature in before if feature not in after_set],
                from_cache=from_cache,
            )
        )

    def _cache_key(self, prefix: str, frame: pd.DataFrame) -> str:
        return f"{prefix}:{hash(tuple(frame.columns))}"

    def _with_protected(self, features: list[str], protected: list[str]) -> list[str]:
        feature_set = set(features)
        combined = list(features)
        for feature in protected:
            if feature not in feature_set:
                combined.append(feature)
                feature_set.add(feature)
        return combined
