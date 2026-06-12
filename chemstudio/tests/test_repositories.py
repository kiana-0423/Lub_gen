from __future__ import annotations

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.models import MoleculeImportRecord
from chemstudio.database.repositories import (
    DescriptorRepository,
    FormulaRepository,
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
