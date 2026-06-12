from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.repositories import DescriptorRepository, ModelRepository
from chemstudio.ml.classifier import train_classification_model
from chemstudio.ml import explainer as shap_explainer
from chemstudio.ml.feature_selection import FeatureSelectionStrategy
from chemstudio.ml.predictor import predict_classification_value, predict_regression_value
from chemstudio.ml.trainer import get_model_catalog, train_regression_model
from chemstudio.services.feature_service import FeatureService
from chemstudio.utils.file_utils import ensure_directory
from chemstudio.validation import validate_feature_names, validate_target_column


logger = logging.getLogger(__name__)


class ModelService:
    """Coordinates training, persistence, and inference for regression models."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        feature_service: FeatureService,
        model_repository: ModelRepository | None = None,
        descriptor_repository: DescriptorRepository | None = None,
    ) -> None:
        """保存训练流程需要的数据库和特征服务依赖。"""
        self.db_manager = db_manager
        self.feature_service = feature_service
        self.model_repository = model_repository or ModelRepository(db_manager)
        self.descriptor_repository = descriptor_repository or DescriptorRepository(db_manager)

    def get_training_dataset(self) -> pd.DataFrame:
        """Return the current merged training dataset with descriptor columns."""
        return self.descriptor_repository.get_wide_dataset(include_mordred=True)

    def get_target_columns(self) -> list[str]:
        """Return candidate property columns for supervised learning."""
        return self.descriptor_repository.list_property_names()

    def get_model_catalog(self, problem_type: str | None = None) -> list[dict[str, Any]]:
        """Return model options for the UI."""
        return get_model_catalog(problem_type)

    def is_explainer_available(self) -> bool:
        """Return whether SHAP explainability is available in this environment."""
        return shap_explainer.is_shap_available()

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
        feature_selection: FeatureSelectionStrategy = "none",
        max_features: int = 150,
    ) -> dict[str, Any]:
        """Train a model using the current database contents."""
        dataset = self.get_training_dataset()
        validate_target_column(dataset, target_name)
        feature_names = self.feature_service.infer_feature_columns(dataset, target_name)
        validate_feature_names(feature_names, list(dataset.columns))
        problem_type = self.infer_problem_type(target_name)
        selected_features, feature_selection_report = self.feature_service.select_features(
            dataset,
            target_name=target_name,
            feature_names=feature_names,
            problem_type=problem_type,
            strategy=feature_selection,
            max_features=max_features,
        )
        validate_feature_names(selected_features, list(dataset.columns))
        train_fn = train_classification_model if problem_type == "classification" else train_regression_model
        artifact = train_fn(
            dataset=dataset,
            target_name=target_name,
            feature_names=selected_features,
            model_key=model_key,
            test_size=test_size,
            cv_mode=cv_mode,
            n_folds=n_folds,
            hp_search=hp_search,
            hp_method=hp_method,
            hp_n_iter=hp_n_iter,
        )
        artifact["feature_selection_report"] = feature_selection_report
        return artifact

    def explain_model(
        self,
        artifact: dict[str, Any],
        x_test: pd.DataFrame | None = None,
    ) -> shap_explainer.SHAPExplanation:
        """Create a SHAP global explanation for a trained model artifact."""
        feature_names = [str(feature) for feature in artifact["feature_names"]]
        x_train_sample = artifact.get("x_train_sample")
        if isinstance(x_train_sample, dict):
            x_train = pd.DataFrame(x_train_sample)
        else:
            dataset = self.get_training_dataset()
            x_train = dataset[feature_names].copy()

        if x_test is None:
            x_test_sample = artifact.get("x_test_sample")
            if isinstance(x_test_sample, dict):
                x_test = pd.DataFrame(x_test_sample)
            else:
                x_test = x_train.head(200).copy()

        return shap_explainer.explain_model(artifact, x_train=x_train, x_test=x_test)

    def explain_single_prediction(
        self,
        artifact: dict[str, Any],
        explanation: shap_explainer.SHAPExplanation,
        feature_values: pd.DataFrame,
    ) -> dict[str, Any]:
        """Create a local SHAP explanation for one prediction input."""
        return shap_explainer.explain_single_prediction(artifact, explanation, feature_values)

    def save_model(self, artifact: dict[str, Any], file_path: str | Path) -> None:
        """Persist a trained model artifact."""
        destination = Path(file_path)
        ensure_directory(destination.parent)
        joblib.dump(artifact, destination)

    def load_model(self, file_path: str | Path, *, trusted_source: bool = False) -> dict[str, Any]:
        """Load a saved model artifact after the caller explicitly trusts the file."""
        source = Path(file_path)
        if not trusted_source:
            raise ValueError("Model files use joblib/pickle and must be loaded only from a trusted source.")
        if source.suffix.lower() != ".joblib":
            raise ValueError("Only .joblib model files are supported.")
        if not source.is_file():
            raise FileNotFoundError(f"Model file does not exist: {source}")

        logger.warning("Loading trusted joblib model artifact from %s", source)
        artifact = joblib.load(source)
        if not isinstance(artifact, dict):
            raise ValueError("Loaded model artifact must be a dictionary.")
        required_keys = {"model", "model_name", "model_key", "target_name", "feature_names"}
        missing_keys = required_keys.difference(artifact)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Loaded model artifact is missing required keys: {missing}")
        if not isinstance(artifact["feature_names"], list):
            raise ValueError("Loaded model artifact has invalid feature_names.")
        return artifact

    def predict(self, artifact: dict[str, Any], feature_values: dict[str, float]) -> float | dict[str, Any]:
        """Predict a property or class using a trained artifact."""
        if artifact.get("problem_type") == "classification":
            return predict_classification_value(artifact, feature_values)
        return predict_regression_value(artifact, feature_values)
