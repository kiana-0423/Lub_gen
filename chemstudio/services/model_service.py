from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from chemstudio.data.db import initialize_database, session_scope
from chemstudio.data.models import ModelRecord, Molecule, Prediction
from chemstudio.data.repositories.molecule_repository import MoleculeRepository
from chemstudio.ml.inference.predictor import Predictor
from chemstudio.ml.trainers.validation import classification_metrics, regression_metrics
from chemstudio.services.descriptor_service import DescriptorService
from chemstudio.utils.paths import model_store_path


MODEL_SPECS = {
    "random_forest_regressor": {"problem_type": "regression", "factory": lambda: RandomForestRegressor(n_estimators=200, random_state=42)},
    "linear_regression": {"problem_type": "regression", "factory": LinearRegression},
    "random_forest_classifier": {"problem_type": "classification", "factory": lambda: RandomForestClassifier(n_estimators=200, random_state=42)},
    "logistic_regression": {"problem_type": "classification", "factory": lambda: LogisticRegression(max_iter=2000, random_state=42)},
}


class ModelService:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url
        self.descriptor_service = DescriptorService()
        self.predictor = Predictor()

    def build_training_dataset(
        self,
        *,
        feature_columns: Sequence[str],
        target_column: str,
        include_hidden: bool = False,
        molecule_ids: Sequence[int] | None = None,
        keyword: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, object]:
        with session_scope(self.database_url) as session:
            repository = MoleculeRepository(session)
            molecules = repository.list_training_candidates(
                include_hidden=include_hidden,
                molecule_ids=molecule_ids,
                keyword=keyword,
                limit=limit,
                offset=offset,
            )

            rows: list[dict[str, object]] = []
            source_molecule_ids: list[int] = []
            for molecule in molecules:
                try:
                    feature_row = {
                        column: self._resolve_value(
                            molecule,
                            column,
                            coerce_numeric=True,
                            session=session,
                            repository=repository,
                        )
                        for column in feature_columns
                    }
                    target_value = self._resolve_value(
                        molecule,
                        target_column,
                        coerce_numeric=False,
                        session=session,
                        repository=repository,
                    )
                except (KeyError, ValueError):
                    continue

                row = dict(feature_row)
                row[target_column] = target_value
                rows.append(row)
                source_molecule_ids.append(molecule.id)

        if not rows:
            raise ValueError("No training rows matched the requested feature/target selection.")

        frame = pd.DataFrame(rows)
        return {
            "dataframe": frame,
            "source_molecule_ids": source_molecule_ids,
            "rows": len(frame.index),
            "feature_columns": list(feature_columns),
            "target_column": target_column,
        }

    def train_model(
        self,
        *,
        name: str,
        model_type: str,
        feature_columns: Sequence[str],
        target_column: str,
        problem_type: str | None = None,
        include_hidden: bool = False,
        molecule_ids: Sequence[int] | None = None,
        keyword: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        save_artifact: bool = True,
    ) -> dict[str, object]:
        initialize_database(self.database_url)
        spec = MODEL_SPECS.get(model_type)
        if spec is None:
            raise ValueError(f"Unsupported model_type: {model_type}")

        resolved_problem_type = problem_type or str(spec["problem_type"])
        if resolved_problem_type != spec["problem_type"]:
            raise ValueError(f"{model_type} does not support problem_type={resolved_problem_type}")

        dataset = self.build_training_dataset(
            feature_columns=feature_columns,
            target_column=target_column,
            include_hidden=include_hidden,
            molecule_ids=molecule_ids,
            keyword=keyword,
            limit=limit,
            offset=offset,
        )
        frame: pd.DataFrame = dataset["dataframe"]
        if len(frame.index) < 2:
            raise ValueError("At least two rows are required for training.")

        features = frame[list(feature_columns)]
        target = frame[target_column]
        estimator = spec["factory"]()

        evaluation = self._fit_and_evaluate(estimator, features, target, resolved_problem_type)
        bundle = {
            "model": evaluation["estimator"],
            "name": name,
            "model_type": model_type,
            "problem_type": resolved_problem_type,
            "feature_columns": list(feature_columns),
            "target_column": target_column,
            "metrics": evaluation["metrics"],
        }

        artifact_path = ""
        if save_artifact:
            artifact_path = str(self._save_bundle(bundle))

        record = self._save_model_record(
            name=name,
            model_type=model_type,
            problem_type=resolved_problem_type,
            target_column=target_column,
            feature_columns=feature_columns,
            metrics=evaluation["metrics"],
            training_config={
                "include_hidden": include_hidden,
                "molecule_ids": list(molecule_ids) if molecule_ids else [],
                "keyword": keyword,
                "limit": limit,
                "offset": offset,
                "rows": dataset["rows"],
                "source_molecule_ids": dataset["source_molecule_ids"],
            },
            artifact_path=artifact_path,
        )

        return {
            "model": self._serialize_model_record(record),
            "metrics": evaluation["metrics"],
            "rows": dataset["rows"],
            "source_molecule_ids": dataset["source_molecule_ids"],
        }

    def load_model_bundle(self, *, model_id: int | None = None, artifact_path: str | None = None) -> dict[str, object]:
        if model_id is None and not artifact_path:
            raise ValueError("model_id or artifact_path is required.")

        if model_id is not None:
            with session_scope(self.database_url) as session:
                record = session.get(ModelRecord, model_id)
                if record is None:
                    raise ValueError(f"Model {model_id} does not exist.")
                artifact_path = record.artifact_path

        bundle = joblib.load(str(artifact_path))
        return bundle

    def list_models(self) -> list[dict[str, object]]:
        with session_scope(self.database_url) as session:
            records = session.query(ModelRecord).order_by(ModelRecord.updated_at.desc(), ModelRecord.id.desc()).all()
            return [self._serialize_model_record(record) for record in records]

    def predict_for_molecules(
        self,
        *,
        model_id: int,
        molecule_ids: Sequence[int],
        save_results: bool = True,
    ) -> dict[str, object]:
        if not molecule_ids:
            return {"predictions": [], "model": None}

        bundle = self.load_model_bundle(model_id=model_id)
        feature_columns = list(bundle["feature_columns"])

        with session_scope(self.database_url) as session:
            repository = MoleculeRepository(session)
            molecules = repository.list_training_candidates(include_hidden=True, molecule_ids=molecule_ids)
            feature_rows: list[dict[str, object]] = []
            selected_ids: list[int] = []
            for molecule in molecules:
                feature_rows.append(
                    {
                        column: self._resolve_value(
                            molecule,
                            column,
                            coerce_numeric=True,
                            session=session,
                            repository=repository,
                        )
                        for column in feature_columns
                    }
                )
                selected_ids.append(molecule.id)

            prediction_payloads = self.predictor.predict(bundle, feature_rows)
            model_record = session.get(ModelRecord, model_id)
            results: list[dict[str, object]] = []
            for molecule_id, payload in zip(selected_ids, prediction_payloads, strict=True):
                prediction = None
                if save_results:
                    prediction = Prediction(
                        model_id=model_id,
                        molecule_id=molecule_id,
                        predicted_value=payload["predicted_value"],
                        predicted_label=payload["predicted_label"],
                        confidence=payload["confidence"],
                        input_features_json=payload["input_features"],
                        metadata_json={"source": "molecule_batch"},
                    )
                    session.add(prediction)
                    session.flush()

                results.append(
                    {
                        "prediction_id": prediction.id if prediction is not None else None,
                        "molecule_id": molecule_id,
                        **payload,
                    }
                )

            return {
                "model": self._serialize_model_record(model_record) if model_record is not None else None,
                "predictions": results,
            }

    def predict_single(
        self,
        *,
        model_id: int,
        molecule_id: int | None = None,
        feature_values: Mapping[str, object] | None = None,
        save_result: bool = True,
    ) -> dict[str, object]:
        bundle = self.load_model_bundle(model_id=model_id)
        feature_columns = list(bundle["feature_columns"])

        if molecule_id is not None:
            result = self.predict_for_molecules(model_id=model_id, molecule_ids=[molecule_id], save_results=save_result)
            predictions = result["predictions"]
            if not predictions:
                raise ValueError(f"Molecule {molecule_id} could not be predicted.")
            return {"model": result["model"], "prediction": predictions[0]}

        if feature_values is None:
            raise ValueError("molecule_id or feature_values is required.")

        row = {column: self._coerce_float(feature_values[column], column) for column in feature_columns}
        prediction_payload = self.predictor.predict(bundle, [row])[0]
        prediction_id = None
        model_snapshot = None
        if save_result:
            with session_scope(self.database_url) as session:
                record = session.get(ModelRecord, model_id)
                if record is None:
                    raise ValueError(f"Model {model_id} does not exist.")
                prediction = Prediction(
                    model_id=model_id,
                    molecule_id=None,
                    predicted_value=prediction_payload["predicted_value"],
                    predicted_label=prediction_payload["predicted_label"],
                    confidence=prediction_payload["confidence"],
                    input_features_json=prediction_payload["input_features"],
                    metadata_json={"source": "feature_payload"},
                )
                session.add(prediction)
                session.flush()
                prediction_id = prediction.id
                model_snapshot = self._serialize_model_record(record)

        return {
            "model": model_snapshot,
            "prediction": {"prediction_id": prediction_id, "molecule_id": None, **prediction_payload},
        }

    def _fit_and_evaluate(self, estimator, features: pd.DataFrame, target: pd.Series, problem_type: str) -> dict[str, object]:
        if len(features.index) >= 5:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=0.25,
                random_state=42,
            )
            estimator.fit(x_train, y_train)
            predictions = estimator.predict(x_test)
            metrics = self._metric_payload(problem_type, y_test, predictions)
            metrics.update(
                {
                    "train_rows": int(len(x_train.index)),
                    "test_rows": int(len(x_test.index)),
                    "feature_count": int(features.shape[1]),
                }
            )
        else:
            estimator.fit(features, target)
            predictions = estimator.predict(features)
            metrics = self._metric_payload(problem_type, target, predictions)
            metrics.update(
                {
                    "train_rows": int(len(features.index)),
                    "test_rows": int(len(features.index)),
                    "feature_count": int(features.shape[1]),
                    "evaluation_mode": "resubstitution",
                }
            )

        return {"estimator": estimator, "metrics": metrics}

    def _metric_payload(self, problem_type: str, y_true, y_pred) -> dict[str, float]:
        if problem_type == "classification":
            return classification_metrics(y_true, y_pred)
        return regression_metrics(y_true, y_pred)

    def _save_bundle(self, bundle: dict[str, object]) -> Path:
        directory = model_store_path()
        directory.mkdir(parents=True, exist_ok=True)
        slug = self._slugify(str(bundle["name"]))
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        path = directory / f"{slug}_{timestamp}.joblib"
        joblib.dump(bundle, path)
        return path

    def _save_model_record(
        self,
        *,
        name: str,
        model_type: str,
        problem_type: str,
        target_column: str,
        feature_columns: Sequence[str],
        metrics: Mapping[str, object],
        training_config: Mapping[str, object],
        artifact_path: str,
    ) -> ModelRecord:
        with session_scope(self.database_url) as session:
            record = ModelRecord(
                name=name,
                model_type=model_type,
                problem_type=problem_type,
                target_name=target_column,
                feature_columns_json=list(feature_columns),
                metrics_json=dict(metrics),
                training_config_json=dict(training_config),
                artifact_path=artifact_path,
            )
            session.add(record)
            session.flush()
            session.refresh(record)
            return record

    def _resolve_value(
        self,
        molecule: Molecule,
        key: str,
        *,
        coerce_numeric: bool,
        session,
        repository: MoleculeRepository,
    ):
        parameter_map = {item.key: item.value for item in molecule.parameters}
        descriptor_map = molecule.descriptor_record.descriptor_values if molecule.descriptor_record else {}

        source: object
        if key.startswith("parameter:"):
            source = parameter_map[key.split(":", 1)[1]]
        elif key.startswith("descriptor:"):
            descriptor_key = key.split(":", 1)[1]
            if descriptor_key not in descriptor_map:
                descriptor_map = repository.save_descriptors(
                    molecule.id,
                    self.descriptor_service.calculate(molecule.canonical_smiles),
                ).descriptor_values
            source = descriptor_map[descriptor_key]
        elif key.startswith("field:"):
            source = getattr(molecule, key.split(":", 1)[1])
        elif key in parameter_map:
            source = parameter_map[key]
        elif key in descriptor_map:
            source = descriptor_map[key]
        elif hasattr(molecule, key):
            source = getattr(molecule, key)
        else:
            raise KeyError(f"Unknown feature or target: {key}")

        if coerce_numeric:
            return self._coerce_float(source, key)
        return source

    def _serialize_model_record(self, record: ModelRecord) -> dict[str, object]:
        return {
            "id": record.id,
            "name": record.name,
            "model_type": record.model_type,
            "problem_type": record.problem_type,
            "target_name": record.target_name,
            "feature_columns": list(record.feature_columns_json or []),
            "metrics": dict(record.metrics_json or {}),
            "training_config": dict(record.training_config_json or {}),
            "artifact_path": record.artifact_path,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }

    @staticmethod
    def _coerce_float(value: object, key: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be numeric, got {value!r}") from exc

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
        return slug or "model"
