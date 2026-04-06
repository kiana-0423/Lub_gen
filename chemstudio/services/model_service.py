from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.ml.predictor import predict_regression_value
from chemstudio.ml.trainer import get_model_catalog, train_regression_model
from chemstudio.services.feature_service import FeatureService
from chemstudio.utils.file_utils import ensure_directory


class ModelService:
    """Coordinates training, persistence, and inference for regression models."""

    def __init__(self, db_manager: DatabaseManager, feature_service: FeatureService) -> None:
        self.db_manager = db_manager
        self.feature_service = feature_service

    def get_training_dataset(self) -> pd.DataFrame:
        """Return the current merged training dataset."""
        return self.db_manager.get_wide_dataset()

    def get_target_columns(self) -> list[str]:
        """Return candidate property columns for supervised learning."""
        return self.db_manager.list_property_names()

    def get_model_catalog(self) -> list[dict[str, Any]]:
        """Return model options for the UI."""
        return get_model_catalog()

    def train_model(self, target_name: str, model_key: str, test_size: float = 0.2) -> dict[str, Any]:
        """Train a regression model using the current database contents."""
        dataset = self.get_training_dataset()
        feature_names = self.feature_service.infer_feature_columns(dataset, target_name)
        return train_regression_model(
            dataset=dataset,
            target_name=target_name,
            feature_names=feature_names,
            model_key=model_key,
            test_size=test_size,
        )

    def save_model(self, artifact: dict[str, Any], file_path: str | Path) -> None:
        """Persist a trained model artifact."""
        destination = Path(file_path)
        ensure_directory(destination.parent)
        joblib.dump(artifact, destination)

    def load_model(self, file_path: str | Path) -> dict[str, Any]:
        """Load a saved model artifact."""
        return joblib.load(file_path)

    def predict(self, artifact: dict[str, Any], feature_values: dict[str, float]) -> float:
        """Predict a scalar property using a trained artifact."""
        return predict_regression_value(artifact, feature_values)
