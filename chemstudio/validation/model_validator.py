from __future__ import annotations

import pandas as pd


def validate_target_column(dataset: pd.DataFrame, target_name: str) -> None:
    """Validate that a target column exists and has at least one value."""
    if target_name not in dataset.columns:
        raise ValueError(f"Target column `{target_name}` was not found in the dataset.")
    if dataset[target_name].dropna().empty:
        raise ValueError(f"Target column `{target_name}` has no usable values.")


def validate_feature_names(feature_names: list[str], dataset_columns: list[str] | None = None) -> None:
    """Validate that a feature-name list is usable and present in the dataset when provided."""
    if not feature_names:
        raise ValueError("No usable feature columns were detected.")
    if any(not str(feature_name).strip() for feature_name in feature_names):
        raise ValueError("Feature names must be non-empty strings.")
    if dataset_columns is None:
        return
    available = set(dataset_columns)
    missing = [feature_name for feature_name in feature_names if feature_name not in available]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Feature columns were not found in the dataset: {joined}")
