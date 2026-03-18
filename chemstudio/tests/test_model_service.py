from __future__ import annotations

from chemstudio.services.model_service import ModelService
from chemstudio.services.molecule_service import MoleculeService


def _seed_training_data(molecule_service: MoleculeService) -> list[dict[str, object]]:
    rows = [
        {"code": "T-001", "name": "ethanol", "smiles": "CCO", "parameters": {"target_score": 46.0, "boiling_point": 78.3}},
        {"code": "T-002", "name": "propanol", "smiles": "CCCO", "parameters": {"target_score": 60.0, "boiling_point": 97.0}},
        {"code": "T-003", "name": "butanol", "smiles": "CCCCO", "parameters": {"target_score": 74.0, "boiling_point": 117.0}},
        {"code": "T-004", "name": "pentanol", "smiles": "CCCCCO", "parameters": {"target_score": 88.0, "boiling_point": 138.0}},
        {"code": "T-005", "name": "hexanol", "smiles": "CCCCCCO", "parameters": {"target_score": 102.0, "boiling_point": 157.0}},
        {"code": "T-006", "name": "heptanol", "smiles": "CCCCCCCO", "parameters": {"target_score": 116.0, "boiling_point": 176.0}, "is_hidden": True},
    ]
    return molecule_service.import_molecules(rows)["items"]


def test_model_service_trains_saves_loads_and_predicts(chemstudio_env):
    molecule_service = MoleculeService()
    seeded = _seed_training_data(molecule_service)

    model_service = ModelService()
    training = model_service.train_model(
        name="target regressor",
        model_type="random_forest_regressor",
        feature_columns=["parameter:molecular_weight", "parameter:boiling_point"],
        target_column="parameter:target_score",
        include_hidden=False,
    )

    assert training["rows"] == 5
    assert training["model"]["artifact_path"]
    assert training["model"]["metrics"]["feature_count"] == 2

    models = model_service.list_models()
    assert len(models) == 1
    assert models[0]["name"] == "target regressor"

    bundle = model_service.load_model_bundle(model_id=training["model"]["id"])
    assert bundle["feature_columns"] == ["parameter:molecular_weight", "parameter:boiling_point"]

    batch_prediction = model_service.predict_for_molecules(
        model_id=training["model"]["id"],
        molecule_ids=[seeded[0]["id"], seeded[1]["id"]],
    )
    assert len(batch_prediction["predictions"]) == 2
    assert batch_prediction["predictions"][0]["prediction_id"] is not None

    single_prediction = model_service.predict_single(
        model_id=training["model"]["id"],
        feature_values={"parameter:molecular_weight": 95.0, "parameter:boiling_point": 150.0},
    )
    assert single_prediction["prediction"]["prediction_id"] is not None
    assert single_prediction["prediction"]["predicted_value"] is not None

    hidden_included = model_service.build_training_dataset(
        feature_columns=["parameter:molecular_weight", "parameter:boiling_point"],
        target_column="parameter:target_score",
        include_hidden=True,
    )
    assert hidden_included["rows"] == 6
