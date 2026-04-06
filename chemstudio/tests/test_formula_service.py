from __future__ import annotations

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.models import MoleculeImportRecord
from chemstudio.services.formula_service import FormulaService


def test_formula_service_saves_lists_and_deletes_formulations(chemstudio_env):
    db_manager = DatabaseManager(chemstudio_env["database_path"])
    db_manager.initialize_database()
    molecule_id = db_manager.insert_molecule_record(MoleculeImportRecord(name="solvent-a", smiles="CCO"))
    service = FormulaService(db_manager)

    record_id = service.save_formulation(
        formula_name="demo-formula",
        note="test note",
        components=[{"molecule_id": molecule_id, "ratio": 100.0, "note": "main"}],
        target_values={"conductivity": "12.5", "capacity": ""},
        test_conditions="temperature=25\npressure=1.0",
    )

    formulations = service.list_formulations()
    assert formulations[0]["id"] == record_id
    assert formulations[0]["formula_name"] == "demo-formula"
    assert formulations[0]["note"] == "test note"
    assert formulations[0]["target_values"]["conductivity"] == 12.5
    assert formulations[0]["components"][0]["name"] == "solvent-a"
    assert formulations[0]["test_conditions"]["condition_temperature"] == 25.0
    assert formulations[0]["test_conditions"]["condition_pressure"] == 1.0

    assert service.delete_formulation(record_id) is True
    assert service.list_formulations() == []


def test_formula_service_auto_normalizes_component_ratios(chemstudio_env):
    db_manager = DatabaseManager(chemstudio_env["database_path"])
    db_manager.initialize_database()
    first_id = db_manager.insert_molecule_record(MoleculeImportRecord(name="mol-a", smiles="CC"))
    second_id = db_manager.insert_molecule_record(MoleculeImportRecord(name="mol-b", smiles="CO"))
    service = FormulaService(db_manager)

    prepared = service.prepare_components(
        [
            {"molecule_id": first_id, "ratio": 2.0},
            {"molecule_id": second_id, "ratio": 3.0},
        ],
        auto_normalize=True,
    )

    ratios = [component["ratio"] for component in prepared["components"]]
    assert round(sum(ratios), 6) == 100.0
    assert round(ratios[0], 4) == 40.0
    assert round(ratios[1], 4) == 60.0


def test_formula_service_trains_and_predicts_from_saved_formulations(chemstudio_env):
    db_manager = DatabaseManager(chemstudio_env["database_path"])
    db_manager.initialize_database()
    first_id = db_manager.insert_molecule_record(MoleculeImportRecord(name="mol-a", smiles="CC"))
    second_id = db_manager.insert_molecule_record(MoleculeImportRecord(name="mol-b", smiles="CO"))
    service = FormulaService(db_manager)

    for ratio in [90.0, 70.0, 50.0, 30.0, 10.0]:
        service.save_formulation(
            formula_name=f"formula-{int(ratio)}",
            note="",
            components=[
                {"molecule_id": first_id, "ratio": ratio},
                {"molecule_id": second_id, "ratio": 100.0 - ratio},
            ],
            target_values={"conductivity": ratio},
            test_conditions={"temperature": 20.0 + ratio / 10.0},
        )

    artifact = service.train_formulation_model(target_name="conductivity", model_key="linear_regression")
    prediction = service.predict_formulation(
        artifact,
        [
            {"molecule_id": first_id, "ratio": 60.0},
            {"molecule_id": second_id, "ratio": 40.0},
        ],
        test_conditions="temperature=26",
    )

    assert artifact["target_name"] == "conductivity"
    assert artifact["feature_names"]
    assert "condition_temperature" in artifact["feature_names"]
    assert prediction["model_name"] == artifact["model_name"]
    assert prediction["test_conditions"]["condition_temperature"] == 26.0
    assert abs(prediction["prediction"] - 60.0) < 15.0
