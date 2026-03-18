from __future__ import annotations

import json

import pandas as pd

from chemstudio.services.import_service import ImportFileService


def test_import_service_loads_json_csv_and_excel(tmp_path):
    service = ImportFileService()

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


def test_import_service_skips_blank_rows_and_rejects_invalid_booleans(tmp_path):
    service = ImportFileService()

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
