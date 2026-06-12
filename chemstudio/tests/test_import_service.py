from __future__ import annotations

import json

import pandas as pd

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.services.data_import_service import DataImportService


def test_data_import_service_loads_json_csv_and_excel(tmp_path):
    service = DataImportService(DatabaseManager(tmp_path / "chemstudio.sqlite"))

    json_path = tmp_path / "molecules.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "code": "J-001",
                    "name": "ethanol",
                    "smiles": "CCO",
                    "parameters": {"family": "alcohol"},
                    "boiling_point": 78.3,
                    "is_hidden": "false",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    json_records = service.load_records(json_path)
    assert json_records[0]["smiles"] == "CCO"
    assert json_records[0]["parameters"]["family"] == "alcohol"
    assert json_records[0]["parameters"]["boiling_point"] == 78.3
    assert json_records[0]["is_hidden"] is False

    csv_path = tmp_path / "molecules.csv"
    pd.DataFrame(
        [
            {
                "code": "C-001",
                "name": "propanol",
                "smiles": "CCCO",
                "parameter:target_score": 60.0,
                "viscosity": 2.1,
                "is_hidden": "yes",
            }
        ]
    ).to_csv(csv_path, index=False)
    csv_records = service.load_records(csv_path)
    assert csv_records[0]["code"] == "C-001"
    assert csv_records[0]["parameters"]["target_score"] == 60.0
    assert csv_records[0]["parameters"]["viscosity"] == 2.1
    assert csv_records[0]["is_hidden"] is True

    xlsx_path = tmp_path / "molecules.xlsx"
    pd.DataFrame(
        [
            {
                "code": "X-001",
                "name": "butanol",
                "canonical_smiles": "CCCCO",
                "parameters": '{"family":"alcohol","target_score":74}',
            }
        ]
    ).to_excel(xlsx_path, index=False)
    xlsx_records = service.load_records(xlsx_path)
    assert xlsx_records[0]["smiles"] == "CCCCO"
    assert xlsx_records[0]["parameters"]["family"] == "alcohol"
    assert xlsx_records[0]["parameters"]["target_score"] == 74


def test_data_import_service_skips_blank_rows_and_rejects_invalid_booleans(tmp_path):
    service = DataImportService(DatabaseManager(tmp_path / "chemstudio.sqlite"))

    csv_path = tmp_path / "blank.csv"
    pd.DataFrame([{"code": None, "name": None, "smiles": None}]).to_csv(csv_path, index=False)
    assert service.load_records(csv_path) == []

    invalid_path = tmp_path / "invalid.csv"
    pd.DataFrame([{"code": "B-001", "smiles": "CCO", "is_hidden": "maybe"}]).to_csv(invalid_path, index=False)
    try:
        service.load_records(invalid_path)
    except ValueError as exc:
        assert "Invalid boolean value" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected invalid boolean import to fail.")


def test_imported_descriptors_are_hidden_from_default_wide_dataset(tmp_path, monkeypatch):
    import chemstudio.services.data_import_service as data_import_module

    monkeypatch.setattr(
        data_import_module,
        "compute_mordred_descriptors",
        lambda smiles: {"ABC": 1.25, "MW": 46.07},
    )

    db_manager = DatabaseManager(tmp_path / "chemstudio.sqlite")
    db_manager.initialize_database()
    service = DataImportService(db_manager)

    csv_path = tmp_path / "molecules.csv"
    pd.DataFrame(
        [
            {
                "name": "ethanol",
                "smiles": "CCO",
                "feature_manual": 3.5,
                "property_viscosity": 1.2,
            }
        ]
    ).to_csv(csv_path, index=False)
    service.import_file(csv_path)

    detail = db_manager.get_molecule_detail(1)
    assert detail is not None
    assert detail.descriptor_values == {"ABC": 1.25, "MW": 46.07}

    public_dataset = db_manager.get_wide_dataset()
    assert "manual" in public_dataset.columns
    assert "ABC" not in public_dataset.columns
    assert "MW" not in public_dataset.columns

    training_dataset = db_manager.get_wide_dataset(include_mordred=True)
    assert float(training_dataset.loc[0, "ABC"]) == 1.25
    assert float(training_dataset.loc[0, "MW"]) == 46.07


def test_import_file_updates_existing_canonical_smiles(tmp_path, monkeypatch):
    import chemstudio.services.data_import_service as data_import_module

    monkeypatch.setattr(
        data_import_module,
        "compute_mordred_descriptors",
        lambda smiles: {"ABC": 2.5},
    )

    db_manager = DatabaseManager(tmp_path / "chemstudio.sqlite")
    db_manager.initialize_database()
    service = DataImportService(db_manager)

    first_path = tmp_path / "first.csv"
    pd.DataFrame(
        [
            {
                "name": "ethanol",
                "smiles": "CCO",
                "feature_manual": 1.0,
                "property_viscosity": 1.2,
            }
        ]
    ).to_csv(first_path, index=False)
    first_result = service.import_file(first_path)

    second_path = tmp_path / "second.csv"
    pd.DataFrame(
        [
            {
                "name": "ethanol_updated",
                "smiles": "CCO",
                "feature_manual": 4.0,
                "property_viscosity": 1.5,
            }
        ]
    ).to_csv(second_path, index=False)
    second_result = service.import_file(second_path)

    assert first_result["inserted_ids"] == second_result["inserted_ids"]
    assert db_manager.count_rows("molecules") == 1
    detail = db_manager.get_molecule_detail(int(first_result["inserted_ids"][0]))
    assert detail is not None
    assert detail.name == "ethanol_updated"
    assert detail.features == {"manual": 4.0}
    assert detail.properties == {"viscosity": 1.5}
    assert detail.descriptor_values == {"ABC": 2.5}


def test_import_file_maps_tribology_schema_and_one_hot_features(tmp_path, monkeypatch):
    import chemstudio.services.data_import_service as data_import_module

    monkeypatch.setattr(
        data_import_module,
        "compute_mordred_descriptors",
        lambda smiles: {"ABC": 2.5},
    )

    db_manager = DatabaseManager(tmp_path / "chemstudio.sqlite")
    db_manager.initialize_database()
    service = DataImportService(db_manager)

    import_path = tmp_path / "tribology.csv"
    pd.DataFrame(
        [
            {
                "编号": "S-001",
                "SMILES": "CCO",
                "分子名称": "ethanol",
                "所采用的基础油": "PAO4",
                "添加浓度": 1.0,
                "test_mode": "SRV",
                "upper_material": "GCr15",
                "upper_type": "ball",
                "lower_material": "GCr15",
                "lower_type": "disk",
                "load_value": 100,
                "Temp": 80,
                "初始氧化温度/℃（PDSC）": 215,
                "平均摩擦系数": 0.088,
                "磨痕宽度/mm": 0.42,
            },
            {
                "编号": "S-002",
                "SMILES": "CCO",
                "分子名称": "ethanol",
                "所采用的基础油": "Ester",
                "添加浓度": 2.0,
                "test_mode": "SRV",
                "upper_material": "GCr15",
                "upper_type": "ball",
                "lower_material": "Steel",
                "lower_type": "plate",
                "load_value": 120,
                "Temp": 90,
                "初始氧化温度/℃（PDSC）": 225,
                "平均摩擦系数": 0.074,
                "磨痕宽度/mm": 0.36,
            },
        ]
    ).to_csv(import_path, index=False)

    result = service.import_file(import_path)

    assert db_manager.count_rows("molecules") == 2
    assert result["inserted_ids"] == [1, 2]
    first_detail = db_manager.get_molecule_detail(1)
    second_detail = db_manager.get_molecule_detail(2)
    assert first_detail is not None
    assert second_detail is not None
    assert first_detail.code == "S-001"
    assert second_detail.code == "S-002"
    assert first_detail.parameters["sample_id"] == "S-001"
    assert first_detail.features["additive_concentration"] == 1.0
    assert first_detail.features["load"] == 100.0
    assert first_detail.features["temperature"] == 80.0
    assert first_detail.features["base_oil__pao4"] == 1.0
    assert first_detail.features["base_oil__ester"] == 0.0
    assert first_detail.features["lower_material__steel"] == 0.0
    assert second_detail.features["lower_material__steel"] == 1.0
    assert first_detail.properties["oxidation_onset_temperature"] == 215.0
    assert first_detail.properties["average_friction_coefficient"] == 0.088
    assert first_detail.properties["wear_scar_width"] == 0.42


def test_compute_missing_descriptors_backfills_existing_molecules(tmp_path, monkeypatch):
    import chemstudio.services.data_import_service as data_import_module

    calls: list[str] = []

    def fake_compute(smiles: str) -> dict[str, float]:
        calls.append(smiles)
        return {"ABC": 1.25}

    monkeypatch.setattr(data_import_module, "compute_mordred_descriptors", fake_compute)

    db_manager = DatabaseManager(tmp_path / "chemstudio.sqlite")
    db_manager.initialize_database()
    saved = db_manager.save_molecule({"name": "ethanol", "smiles": "CCO", "canonical_smiles": "CCO"})
    service = DataImportService(db_manager)

    result = service.compute_missing_descriptors([int(saved["id"])])

    assert result["computed_count"] == 1
    assert result["skipped_count"] == 0
    assert calls == ["CCO"]
    detail = db_manager.get_molecule_detail(int(saved["id"]))
    assert detail is not None
    assert detail.descriptor_values == {"ABC": 1.25}

    second_result = service.compute_missing_descriptors([int(saved["id"])])
    assert second_result["computed_count"] == 0
    assert second_result["skipped_count"] == 1
    assert calls == ["CCO"]


def test_data_import_service_parallelizes_large_descriptor_batches(tmp_path, monkeypatch):
    import chemstudio.services.data_import_service as data_import_module

    calls: list[str] = []

    class FakeParallel:
        def __init__(self, *, n_jobs, prefer):
            assert n_jobs == -1
            assert prefer == "threads"

        def __call__(self, tasks):
            return [task() for task in tasks]

    def fake_delayed(function):
        def build_task(smiles):
            return lambda: function(smiles)

        return build_task

    def fake_compute(smiles):
        calls.append(smiles)
        return {"descriptor": float(len(smiles))}

    monkeypatch.setattr(data_import_module, "Parallel", FakeParallel)
    monkeypatch.setattr(data_import_module, "delayed", fake_delayed)
    monkeypatch.setattr(data_import_module, "compute_mordred_descriptors", fake_compute)

    service = DataImportService(DatabaseManager(tmp_path / "chemstudio.sqlite"))
    results = service._compute_descriptors_parallel(["CCO"] * 50)

    assert len(results) == 50
    assert calls == ["CCO"] * 50
    assert results[0] == {"descriptor": 3.0}
