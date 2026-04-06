from __future__ import annotations

import json
from typing import Any

import pandas as pd

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.ml.predictor import predict_regression_value
from chemstudio.ml.trainer import get_model_catalog, train_regression_model
from chemstudio.utils.file_utils import normalize_field_name, parse_feature_text


class FormulaService:
    """Manages formulation records, feature building, training, and prediction."""

    DEFAULT_TARGET_FIELDS = ["conductivity", "capacity", "viscosity", "stability"]

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def get_model_catalog(self) -> list[dict[str, Any]]:
        """Return model options for the formulation-training UI."""
        return get_model_catalog()

    def get_available_target_fields(self) -> list[str]:
        """Return default and previously-saved formulation target names."""
        target_fields = set(self.DEFAULT_TARGET_FIELDS)
        for record in self.list_formulations(limit=1000):
            target_fields.update(record["target_values"].keys())
        return sorted(target_fields)

    def validate_target_values(self, raw_target_values: dict[str, str | float | int | None]) -> dict[str, float]:
        """Normalize target inputs and discard empty fields."""
        normalized: dict[str, float] = {}
        for field_name, raw_value in raw_target_values.items():
            if raw_value is None:
                continue
            if isinstance(raw_value, str):
                stripped = raw_value.strip()
                if not stripped:
                    continue
                try:
                    normalized[field_name] = float(stripped)
                except ValueError as exc:
                    raise ValueError(f"目标字段 `{field_name}` 需要是数值。") from exc
                continue
            normalized[field_name] = float(raw_value)
        return normalized

    def parse_test_conditions(self, raw_conditions: str | dict[str, Any] | None) -> dict[str, float]:
        """Parse optional test-condition inputs from JSON or key=value text."""
        if raw_conditions is None:
            return {}
        if isinstance(raw_conditions, dict):
            normalized_conditions: dict[str, float] = {}
            for key, value in raw_conditions.items():
                normalized_key = normalize_field_name(key)
                if not normalized_key.startswith("condition_"):
                    normalized_key = f"condition_{normalized_key}"
                normalized_conditions[normalized_key] = float(value)
            return normalized_conditions
        parsed = parse_feature_text(raw_conditions)
        return {f"condition_{normalize_field_name(key)}": float(value) for key, value in parsed.items()}

    def prepare_components(
        self,
        components: list[dict[str, Any]],
        *,
        auto_normalize: bool = False,
    ) -> dict[str, Any]:
        """Validate components, attach molecule metadata, and normalize ratios when requested."""
        if not components:
            raise ValueError("至少需要一个配方组分。")

        prepared: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        total_ratio = 0.0

        for index, component in enumerate(components, start=1):
            molecule_id_raw = component.get("molecule_id")
            if molecule_id_raw is None:
                raise ValueError(f"第 {index} 行未选择分子。")

            try:
                molecule_id = int(molecule_id_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {index} 行分子 ID 无效。") from exc

            if molecule_id in seen_ids:
                raise ValueError("同一个分子在单个配方中只能出现一次。")
            seen_ids.add(molecule_id)

            ratio_raw = component.get("ratio")
            try:
                ratio = float(ratio_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {index} 行比例必须是数值。") from exc

            if ratio < 0:
                raise ValueError(f"第 {index} 行比例不能为负数。")

            detail = self.db_manager.get_molecule_detail(molecule_id)
            if detail is None:
                raise ValueError(f"分子 ID {molecule_id} 不存在。")

            note = str(component.get("note") or "").strip()
            prepared.append(
                {
                    "molecule_id": molecule_id,
                    "name": detail.name,
                    "smiles": detail.smiles,
                    "ratio": ratio,
                    "note": note,
                }
            )
            total_ratio += ratio

        if total_ratio <= 0:
            raise ValueError("比例总和必须大于 0。")

        normalized = False
        scale = 1.0
        if abs(total_ratio - 100.0) <= 1e-3:
            scale = 1.0
        elif abs(total_ratio - 1.0) <= 1e-6:
            scale = 100.0
            normalized = True
        elif auto_normalize:
            scale = 100.0 / total_ratio
            normalized = True
        else:
            raise ValueError(f"比例总和当前为 {total_ratio:.4f}，需要等于 100。")

        if scale != 1.0:
            for component in prepared:
                component["ratio"] = float(component["ratio"]) * scale

        return {
            "components": prepared,
            "ratio_sum": total_ratio if not normalized else 100.0,
            "normalized": normalized,
            "normalized_sum": sum(float(component["ratio"]) for component in prepared),
        }

    def save_formulation(
        self,
        *,
        formula_name: str,
        note: str,
        components: list[dict[str, Any]],
        target_values: dict[str, str | float | int | None],
        test_conditions: str | dict[str, Any] | None = None,
        auto_normalize: bool = False,
    ) -> int:
        """Persist a formulation record for later browsing and model training."""
        normalized_name = formula_name.strip() or "unnamed_formula"
        prepared = self.prepare_components(components, auto_normalize=auto_normalize)
        normalized_targets = self.validate_target_values(target_values)
        parsed_conditions = self.parse_test_conditions(test_conditions)
        return self.db_manager.save_formulation(
            formula_name=normalized_name,
            note=note.strip(),
            composition=prepared["components"],
            conditions=parsed_conditions,
            target_values=normalized_targets,
        )

    def list_formulations(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return parsed formulation rows for the formula-design module."""
        records: list[dict[str, Any]] = []
        for row in self.db_manager.list_formulations(limit=limit):
            records.append(self._deserialize_formulation_row(row))
        return records

    def get_formulation_detail(self, formulation_id: int) -> dict[str, Any] | None:
        """Return one parsed formulation record."""
        row = self.db_manager.get_formulation(formulation_id)
        if row is None:
            return None
        return self._deserialize_formulation_row(row)

    def delete_formulation(self, formulation_id: int) -> bool:
        """Delete a formulation record."""
        return self.db_manager.delete_formulation(formulation_id)

    def build_ratio_feature_vector(
        self,
        components: list[dict[str, Any]],
        *,
        feature_names: list[str] | None = None,
        test_conditions: str | dict[str, Any] | None = None,
        auto_normalize: bool = False,
    ) -> dict[str, Any]:
        """Convert a formulation into a simple ratio-vector feature representation."""
        prepared = self.prepare_components(components, auto_normalize=auto_normalize)
        prepared_components = prepared["components"]

        raw_features = {
            f"molecule_{int(component['molecule_id'])}": float(component["ratio"]) / 100.0
            for component in prepared_components
        }
        raw_features.update(self.parse_test_conditions(test_conditions))
        aligned_features = (
            {feature_name: float(raw_features.get(feature_name, 0.0)) for feature_name in feature_names}
            if feature_names is not None
            else raw_features
        )

        return {
            "features": aligned_features,
            "components": prepared_components,
            "ratio_sum": float(sum(float(component["ratio"]) for component in prepared_components)),
        }

    def build_training_dataset(self, target_name: str) -> tuple[pd.DataFrame, list[str]]:
        """Build a formulation-level dataset using ratio vectors and a selected target field."""
        samples: list[dict[str, float | int]] = []
        feature_names: set[str] = set()

        for record in self.list_formulations(limit=5000):
            target_values = record["target_values"]
            if target_name not in target_values:
                continue

            feature_payload = self.build_ratio_feature_vector(
                record["components"],
                test_conditions=record["test_conditions"],
                auto_normalize=True,
            )
            sample: dict[str, float | int] = {
                "formulation_id": int(record["id"]),
                target_name: float(target_values[target_name]),
            }
            for feature_name, feature_value in feature_payload["features"].items():
                sample[feature_name] = float(feature_value)
            feature_names.update(feature_payload["features"].keys())
            samples.append(sample)

        if not samples:
            raise ValueError(f"没有找到带有目标字段 `{target_name}` 的配方样本。")

        dataset = pd.DataFrame(samples)
        ordered_features = sorted(feature_names)
        for feature_name in ordered_features:
            if feature_name not in dataset.columns:
                dataset[feature_name] = 0.0

        return dataset.fillna(0.0), ordered_features

    def train_formulation_model(
        self,
        *,
        target_name: str,
        model_key: str,
        test_size: float = 0.25,
    ) -> dict[str, Any]:
        """Train a formulation regression model from saved formulation records."""
        dataset, feature_names = self.build_training_dataset(target_name)
        sample_count = len(dataset)
        effective_test_size = max(float(test_size), min(0.5, 2.0 / max(sample_count, 1)))
        artifact = train_regression_model(
            dataset=dataset,
            target_name=target_name,
            feature_names=feature_names,
            model_key=model_key,
            test_size=effective_test_size,
        )
        artifact["feature_space"] = "formulation_ratio_vector"
        return artifact

    def predict_formulation(
        self,
        artifact: dict[str, Any],
        components: list[dict[str, Any]],
        *,
        test_conditions: str | dict[str, Any] | None = None,
        auto_normalize: bool = False,
    ) -> dict[str, Any]:
        """Predict a formulation property using the current trained artifact."""
        payload = self.build_ratio_feature_vector(
            components,
            feature_names=list(artifact["feature_names"]),
            test_conditions=test_conditions,
            auto_normalize=auto_normalize,
        )
        prediction = predict_regression_value(artifact, payload["features"])
        return {
            "target_name": str(artifact["target_name"]),
            "model_name": str(artifact["model_name"]),
            "prediction": prediction,
            "features": payload["features"],
            "components": payload["components"],
            "test_conditions": self.parse_test_conditions(test_conditions),
        }

    def predict_formula(self, artifact: dict[str, Any], components: list[dict[str, Any]]) -> dict[str, Any]:
        """Backward-compatible wrapper for formula prediction."""
        return self.predict_formulation(artifact, components, auto_normalize=False)

    def save_formula_result(self, formula_name: str, prediction_result: dict[str, Any]) -> int:
        """Backward-compatible wrapper for saving a predicted formula result."""
        predicted_properties = {str(prediction_result["target_name"]): float(prediction_result["prediction"])}
        return self.db_manager.save_formula(
            formula_name=formula_name,
            composition=list(prediction_result["components"]),
            conditions=dict(prediction_result.get("test_conditions") or {}),
            predicted_properties=predicted_properties,
        )

    def _deserialize_formulation_row(self, row: dict[str, Any]) -> dict[str, Any]:
        composition = self._safe_json_loads(row.get("composition_json"), default=[])
        conditions = self._safe_json_loads(row.get("conditions_json"), default={})
        predicted_properties = self._safe_json_loads(row.get("predicted_property_json"), default={})
        normalized_components = list(composition)
        if isinstance(composition, list):
            try:
                normalized_components = self.prepare_components(list(composition), auto_normalize=True)["components"]
            except ValueError:
                normalized_components = list(composition)
        return {
            "id": int(row["id"]),
            "formula_name": str(row["formula_name"]),
            "note": str(row.get("note") or ""),
            "components": normalized_components,
            "test_conditions": {str(key): float(value) for key, value in dict(conditions).items()},
            "target_values": {str(key): float(value) for key, value in dict(predicted_properties).items()},
            "created_at": str(row["created_at"]),
        }

    def _safe_json_loads(self, raw_value: Any, *, default: Any) -> Any:
        if raw_value in {None, ""}:
            return default
        try:
            return json.loads(str(raw_value))
        except (TypeError, json.JSONDecodeError):
            return default
