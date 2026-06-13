from __future__ import annotations

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.models import MoleculeImportRecord
from chemstudio.database.repositories import (
    DescriptorRepository,
    FormulaRepository,
    MaterialRepository,
    ModelRepository,
    PredictionRepository,
)


def test_repository_facades_cover_non_molecule_tables(tmp_path):
    db_manager = DatabaseManager(tmp_path / "chemstudio.sqlite")
    db_manager.initialize_database()
    molecule_id = db_manager.insert_molecule_record(
        MoleculeImportRecord(
            name="ethanol",
            smiles="CCO",
            features={"manual_feature": 1.5},
            properties={"viscosity": 1.2},
        )
    )

    descriptor_repository = DescriptorRepository(db_manager)
    descriptor_repository.save_descriptors(molecule_id, {"ABC": 2.5})
    assert descriptor_repository.list_feature_names() == ["manual_feature"]
    assert descriptor_repository.list_property_names() == ["viscosity"]
    wide_dataset = descriptor_repository.get_wide_dataset(include_mordred=True)
    assert float(wide_dataset.loc[0, "ABC"]) == 2.5
    assert float(wide_dataset.loc[0, "manual_feature"]) == 1.5

    model_repository = ModelRepository(db_manager)
    model_id = model_repository.save_model_record(
        name="viscosity_model",
        model_type="LinearRegression",
        problem_type="regression",
        target_name="viscosity",
        feature_columns=["manual_feature"],
        metrics={"r2": 1.0},
    )
    assert model_id > 0
    assert model_repository.list_model_records()[0]["name"] == "viscosity_model"

    prediction_repository = PredictionRepository(db_manager)
    prediction_id = prediction_repository.save_prediction_record(
        model_id=model_id,
        molecule_id=molecule_id,
        predicted_value=1.1,
        input_features={"manual_feature": 1.5},
    )
    assert prediction_id > 0

    formula_repository = FormulaRepository(db_manager)
    formulation_id = formula_repository.save_formulation(
        formula_name="ethanol_formula",
        note="",
        composition=[{"molecule_id": molecule_id, "ratio": 100.0}],
        target_values={"viscosity": 1.2},
    )
    assert formulation_id > 0
    assert formula_repository.list_formulations()[0]["formula_name"] == "ethanol_formula"
    assert formula_repository.get_formulation(formulation_id) is not None
    assert formula_repository.delete_formulation(formulation_id) is True


def test_material_repository_persists_lubricant_metadata(tmp_path):
    db_manager = DatabaseManager(tmp_path / "chemstudio.sqlite")
    db_manager.initialize_database()
    base_oil_id = db_manager.insert_molecule_record(
        MoleculeImportRecord(name="base-oil", smiles="CCCC", material_type_id=1)
    )
    additive_id = db_manager.insert_molecule_record(MoleculeImportRecord(name="additive", smiles="CCO"))

    material_repository = MaterialRepository(db_manager)
    material_types = material_repository.list_material_types()
    assert len(material_types) >= 23
    assert {row["type_name"] for row in material_types} == {"additive", "base_oil"}

    property_id = material_repository.save_lubricant_property(
        additive_id,
        "weld_load",
        1234.0,
        property_unit="N",
        test_standard="ASTM D2783",
    )
    assert property_id > 0
    properties = material_repository.get_lubricant_properties(additive_id)
    assert properties[0]["property_name"] == "weld_load"
    assert float(properties[0]["property_value"]) == 1234.0

    compatibility_id = material_repository.save_additive_compatibility(
        additive_id,
        base_oil_id,
        compatibility_score=0.85,
        solubility="good",
    )
    assert compatibility_id > 0
    compatibilities = material_repository.get_additive_compatibilities(additive_id=additive_id)
    assert compatibilities[0]["additive_name"] == "additive"
    assert compatibilities[0]["base_oil_name"] == "base-oil"


def test_formula_repository_saves_component_details(tmp_path):
    db_manager = DatabaseManager(tmp_path / "chemstudio.sqlite")
    db_manager.initialize_database()
    molecule_id = db_manager.insert_molecule_record(MoleculeImportRecord(name="component-a", smiles="CCO"))

    formula_repository = FormulaRepository(db_manager)
    formula_id = formula_repository.save_formulation(
        formula_name="blend-a",
        note="",
        composition=[{"molecule_id": molecule_id, "ratio": 10.0}],
        components=[{"molecule_id": molecule_id, "component_role": "additive", "ratio": 10.0, "sort_order": 1}],
        target_values={"wear_scar_width": 0.3},
    )

    detail = formula_repository.get_formula_detail(formula_id)
    assert detail is not None
    assert detail["components"][0]["name"] == "component-a"
    assert float(detail["components"][0]["ratio"]) == 10.0
