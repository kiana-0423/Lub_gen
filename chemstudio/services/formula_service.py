from __future__ import annotations

from typing import Any

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.ml.predictor import predict_regression_value


class FormulaService:
    """Builds formula-level features and runs formula property prediction."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def build_formula_features(
        self,
        components: list[dict[str, float | int | str]],
        required_features: list[str] | None = None,
    ) -> dict[str, Any]:
        """Aggregate molecule features into a simple weighted-average formula vector."""
        if not components:
            raise ValueError("At least one formula component is required.")

        total_ratio = sum(float(component["ratio"]) for component in components)
        if abs(total_ratio - 100.0) <= 1e-6:
            divisor = 100.0
        elif abs(total_ratio - 1.0) <= 1e-6:
            divisor = 1.0
        else:
            raise ValueError("Component ratios must sum to 100 or 1.0.")

        details: list[dict[str, Any]] = []
        aggregated: dict[str, float] = {feature_name: 0.0 for feature_name in (required_features or [])}

        for component in components:
            molecule_id = int(component["molecule_id"])
            ratio = float(component["ratio"]) / divisor
            detail = self.db_manager.get_molecule_detail(molecule_id)
            if detail is None:
                raise ValueError(f"Molecule {molecule_id} was not found.")

            details.append(
                {
                    "molecule_id": molecule_id,
                    "name": detail.name,
                    "smiles": detail.smiles,
                    "ratio": ratio,
                }
            )

            feature_source = detail.features
            if required_features is None:
                for feature_name, feature_value in feature_source.items():
                    aggregated[feature_name] = aggregated.get(feature_name, 0.0) + float(feature_value) * ratio
            else:
                for feature_name in required_features:
                    aggregated[feature_name] = aggregated.get(feature_name, 0.0) + float(feature_source.get(feature_name, 0.0)) * ratio

        return {"features": aggregated, "components": details}

    def predict_formula(self, artifact: dict[str, Any], components: list[dict[str, float | int | str]]) -> dict[str, Any]:
        """Generate formula features and predict the target property."""
        payload = self.build_formula_features(components, required_features=list(artifact["feature_names"]))
        prediction = predict_regression_value(artifact, payload["features"])
        return {
            "target_name": artifact["target_name"],
            "prediction": prediction,
            "features": payload["features"],
            "components": payload["components"],
        }

    def save_formula_result(self, formula_name: str, prediction_result: dict[str, Any]) -> int:
        """Persist a formula prediction result."""
        predicted_properties = {str(prediction_result["target_name"]): float(prediction_result["prediction"])}
        return self.db_manager.save_formula(formula_name, prediction_result["components"], predicted_properties)
