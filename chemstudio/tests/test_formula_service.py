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
    assert formulations[0]["components"][0]["component_role"] == "additive"
    assert formulations[0]["test_conditions"]["condition_temperature"] == 25.0
    assert formulations[0]["test_conditions"]["condition_pressure"] == 1.0
    assert db_manager.get_formula_components(record_id)[0]["component_role"] == "additive"

    assert service.delete_formulation(record_id) is True
    assert service.list_formulations() == []


def test_formula_service_preserves_component_roles_and_role_features(chemstudio_env):
    db_manager = DatabaseManager(chemstudio_env["database_path"])
    db_manager.initialize_database()
    base_type_id = int(db_manager.list_material_types("base_oil")[0]["id"])
    additive_type_id = next(
        int(row["id"])
        for row in db_manager.list_material_types("additive")
        if row["category"] == "antioxidant"
    )
    base_id = db_manager.insert_molecule_record(
        MoleculeImportRecord(name="base", smiles="CCCC", material_type_id=base_type_id)
    )
    additive_id = db_manager.insert_molecule_record(
        MoleculeImportRecord(name="antioxidant", smiles="CCO", material_type_id=additive_type_id)
    )
    service = FormulaService(db_manager)

    record_id = service.save_formulation(
        formula_name="role-formula",
        note="",
        components=[
            {"molecule_id": base_id, "component_role": "base_oil", "ratio": 90.0},
            {"molecule_id": additive_id, "component_role": "additive", "ratio": 10.0},
        ],
        target_values={"viscosity": 1.0},
    )

    detail = service.get_formulation_detail(record_id)
    assert detail is not None
    assert [component["component_role"] for component in detail["components"]] == ["base_oil", "additive"]
    feature_payload = service.build_ratio_feature_vector(detail["components"], auto_normalize=True)
    assert feature_payload["features"]["base_oil_total_ratio"] == 0.9
    assert feature_payload["features"]["additive_total_ratio"] == 0.1
    assert feature_payload["features"]["additive_count"] == 1.0
    assert feature_payload["features"]["antioxidant_ratio"] == 0.1


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
