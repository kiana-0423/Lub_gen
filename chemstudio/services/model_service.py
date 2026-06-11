from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.ml.classifier import train_classification_model
from chemstudio.ml.predictor import predict_classification_value, predict_regression_value
from chemstudio.ml.trainer import get_model_catalog, train_regression_model
from chemstudio.services.feature_service import FeatureService
from chemstudio.utils.file_utils import ensure_directory


class ModelService:
    """Coordinates training, persistence, and inference for regression models."""

    def __init__(self, db_manager: DatabaseManager, feature_service: FeatureService) -> None:
        """保存训练流程需要的数据库和特征服务依赖。"""
        self.db_manager = db_manager
        self.feature_service = feature_service

    def get_training_dataset(self) -> pd.DataFrame:
        """Return the current merged training dataset."""
        return self.db_manager.get_wide_dataset()

    def get_target_columns(self) -> list[str]:
        """Return candidate property columns for supervised learning."""
        return self.db_manager.list_property_names()

    def get_model_catalog(self, problem_type: str | None = None) -> list[dict[str, Any]]:
        """Return model options for the UI."""
        return get_model_catalog(problem_type)

    def infer_problem_type(self, target_name: str) -> str:
        """Infer classification when the target has <= 10 integer-valued classes."""
        dataset = self.get_training_dataset()
        if target_name not in dataset.columns:
            raise ValueError(f"Target column `{target_name}` was not found in the dataset.")
        target_values = pd.to_numeric(dataset[target_name], errors="coerce").dropna()
        if target_values.empty:
            return "regression"
        unique_values = target_values.unique()
        all_integer = all(float(value).is_integer() for value in unique_values)
        if len(unique_values) <= 10 and all_integer:
            return "classification"
        return "regression"

    def train_model(
        self,
        target_name: str,
        model_key: str,
        test_size: float = 0.2,
        cv_mode: bool = False,
        n_folds: int = 5,
        hp_search: bool = False,
        hp_method: str = "grid",
        hp_n_iter: int = 20,
    ) -> dict[str, Any]:
        """Train a model using the current database contents."""
        dataset = self.get_training_dataset()
        feature_names = self.feature_service.infer_feature_columns(dataset, target_name)
        problem_type = self.infer_problem_type(target_name)
        train_fn = train_classification_model if problem_type == "classification" else train_regression_model
        return train_fn(
            dataset=dataset,
            target_name=target_name,
            feature_names=feature_names,
            model_key=model_key,
            test_size=test_size,
            cv_mode=cv_mode,
            n_folds=n_folds,
            hp_search=hp_search,
            hp_method=hp_method,
            hp_n_iter=hp_n_iter,
        )

    def save_model(self, artifact: dict[str, Any], file_path: str | Path) -> None:
        """Persist a trained model artifact."""
        destination = Path(file_path)
        ensure_directory(destination.parent)
        joblib.dump(artifact, destination)

    def load_model(self, file_path: str | Path) -> dict[str, Any]:
        """Load a saved model artifact."""
        return joblib.load(file_path)

    def predict(self, artifact: dict[str, Any], feature_values: dict[str, float]) -> float | dict[str, Any]:
        """Predict a property or class using a trained artifact."""
        if artifact.get("problem_type") == "classification":
            return predict_classification_value(artifact, feature_values)
        return predict_regression_value(artifact, feature_values)
