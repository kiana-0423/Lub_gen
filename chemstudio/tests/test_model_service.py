from __future__ import annotations

import pandas as pd

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.services.data_import_service import DataImportService
from chemstudio.services.feature_service import FeatureService
from chemstudio.services.formula_service import FormulaService
from chemstudio.services.model_service import ModelService


def _seed_current_dataset(db_manager: DatabaseManager, tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "name": "water",
                "smiles": "O",
                "source": "test",
                "feature_mol_wt": 18.02,
                "feature_mol_logp": -1.38,
                "feature_tpsa": 31.50,
                "property_capacity": 88.0,
                "property_phase": 0,
                "property_viscosity": 1.0,
            },
            {
                "name": "ethanol",
                "smiles": "CCO",
                "source": "test",
                "feature_mol_wt": 46.07,
                "feature_mol_logp": -0.31,
                "feature_tpsa": 20.23,
                "property_capacity": 102.0,
                "property_phase": 1,
                "property_viscosity": 1.2,
            },
            {
                "name": "acetonitrile",
                "smiles": "CC#N",
                "source": "test",
                "feature_mol_wt": 41.05,
                "feature_mol_logp": -0.34,
                "feature_tpsa": 23.79,
                "property_capacity": 98.0,
                "property_phase": 0,
                "property_viscosity": 0.45,
            },
            {
                "name": "dimethyl carbonate",
                "smiles": "COC(=O)OC",
                "source": "test",
                "feature_mol_wt": 90.08,
                "feature_mol_logp": 0.20,
                "feature_tpsa": 35.53,
                "property_capacity": 105.0,
                "property_phase": 1,
                "property_viscosity": 0.60,
            },
            {
                "name": "ethylene carbonate",
                "smiles": "O=C1OCCO1",
                "source": "test",
                "feature_mol_wt": 88.06,
                "feature_mol_logp": -0.20,
                "feature_tpsa": 35.53,
                "property_capacity": 120.0,
                "property_phase": 1,
                "property_viscosity": 1.92,
            },
            {
                "name": "propylene carbonate",
                "smiles": "CC1COC(=O)O1",
                "source": "test",
                "feature_mol_wt": 102.09,
                "feature_mol_logp": 0.10,
                "feature_tpsa": 35.53,
                "property_capacity": 118.0,
                "property_phase": 0,
                "property_viscosity": 2.50,
            },
        ]
    )
    csv_path = tmp_path / "training.csv"
    frame.to_csv(csv_path, index=False)
    DataImportService(db_manager).import_file(csv_path)


def test_model_service_trains_saves_loads_predicts_and_supports_formulas(tmp_path):
    db_manager = DatabaseManager(tmp_path / "chemstudio.sqlite")
    db_manager.initialize_database()
    _seed_current_dataset(db_manager, tmp_path)

    feature_service = FeatureService(db_manager)
    model_service = ModelService(db_manager, feature_service)

    dataset = model_service.get_training_dataset()
    assert len(dataset) == 6
    assert "capacity" in model_service.get_target_columns()
    assert model_service.infer_problem_type("phase") == "classification"
    assert model_service.infer_problem_type("viscosity") == "regression"

    artifact = model_service.train_model(
        target_name="viscosity",
        model_key="random_forest",
        test_size=0.33,
        cv_mode=True,
        n_folds=3,
    )
    assert artifact["model_name"] == "RandomForestRegressor"
    assert artifact["problem_type"] == "regression"
    assert artifact["target_name"] == "viscosity"
    assert len(artifact["feature_names"]) == 3
    assert len(artifact["y_true"]) == len(artifact["y_pred"])
    assert artifact["cv_results"]["n_folds"] == 3

    classification_artifact = model_service.train_model(
        target_name="phase",
        model_key="random_forest_classifier",
        test_size=0.33,
    )
    assert classification_artifact["problem_type"] == "classification"
    assert "accuracy" in classification_artifact["metrics"]
    classification_prediction = model_service.predict(
        classification_artifact,
        {"mol_wt": 80.0, "mol_logp": 0.0, "tpsa": 30.0},
    )
    assert isinstance(classification_prediction, dict)
    assert "label" in classification_prediction
    assert "probabilities" in classification_prediction

    save_path = tmp_path / "viscosity_model.joblib"
    model_service.save_model(artifact, save_path)
    loaded_artifact = model_service.load_model(save_path)
    assert loaded_artifact["feature_names"] == artifact["feature_names"]

    prediction = model_service.predict(
        loaded_artifact,
        {"mol_wt": 80.0, "mol_logp": 0.0, "tpsa": 30.0},
    )
    assert isinstance(prediction, float)

    formula_service = FormulaService(db_manager)
    formula_prediction = formula_service.predict_formula(
        loaded_artifact,
        [
            {"molecule_id": 1, "ratio": 50.0},
            {"molecule_id": 2, "ratio": 50.0},
        ],
    )
    assert formula_prediction["target_name"] == "viscosity"
    assert len(formula_prediction["components"]) == 2

    record_id = formula_service.save_formula_result("water_ethanol", formula_prediction)
    assert record_id > 0
    assert len(db_manager.list_formulas()) == 1
