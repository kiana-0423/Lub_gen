from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd


class ImportFileService:
    RESERVED_FIELDS = {
        "code",
        "name",
        "smiles",
        "canonical_smiles",
        "is_hidden",
        "notes",
        "molblock",
        "parameters",
    }

    def load_records(self, file_path: str | Path) -> list[dict[str, object]]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            raw_records = self._load_json(path)
        elif suffix == ".csv":
            raw_records = self._load_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            raw_records = self._load_excel(path)
        else:
            raise ValueError(f"Unsupported import format: {path.suffix or '<none>'}")

        records: list[dict[str, object]] = []
        for index, raw_record in enumerate(raw_records, start=1):
            normalized = self._normalize_record(raw_record, row_index=index)
            if normalized is not None:
                records.append(normalized)
        return records

    def _load_json(self, path: Path) -> list[Mapping[str, object]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
            records = payload["items"]
        else:
            raise ValueError("JSON import expects a list of records or an object with an 'items' list.")

        self._validate_sequence(records)
        return list(records)

    def _load_csv(self, path: Path) -> list[dict[str, object]]:
        frame = pd.read_csv(path)
        return self._frame_to_records(frame)

    def _load_excel(self, path: Path) -> list[dict[str, object]]:
        frame = pd.read_excel(path)
        return self._frame_to_records(frame)

    def _frame_to_records(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        normalized_frame = frame.where(pd.notna(frame), None)
        return list(normalized_frame.to_dict(orient="records"))

    def _normalize_record(self, raw_record: Mapping[str, object], *, row_index: int) -> dict[str, object] | None:
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"Row {row_index} must be an object.")

        row = {str(key): self._normalize_cell(value) for key, value in raw_record.items()}

        parameters = self._parse_parameters(row.get("parameters"), row_index=row_index)
        for key, value in row.items():
            if key in self.RESERVED_FIELDS:
                continue
            if value is None or value == "":
                continue
            parameter_key = key.split("parameter:", 1)[1] if key.startswith("parameter:") else key
            parameters[str(parameter_key)] = value

        smiles = self._clean_text(row.get("smiles")) or self._clean_text(row.get("canonical_smiles"))
        if not self._has_meaningful_data(row, parameters):
            return None
        if not smiles:
            raise ValueError(f"Row {row_index} is missing a SMILES value.")

        return {
            "code": self._clean_text(row.get("code")),
            "name": self._clean_text(row.get("name")) or "",
            "smiles": smiles,
            "is_hidden": self._parse_bool(row.get("is_hidden")),
            "notes": self._clean_text(row.get("notes")) or "",
            "molblock": self._clean_text(row.get("molblock")) or "",
            "parameters": parameters,
        }

    def _parse_parameters(self, raw_value: object, *, row_index: int) -> dict[str, object]:
        if raw_value in (None, ""):
            return {}

        if isinstance(raw_value, Mapping):
            return {str(key): self._normalize_cell(value) for key, value in raw_value.items()}

        if isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Row {row_index} has invalid parameters JSON: {exc.msg}") from exc
            if not isinstance(parsed, Mapping):
                raise ValueError(f"Row {row_index} parameters must decode to an object.")
            return {str(key): self._normalize_cell(value) for key, value in parsed.items()}

        raise ValueError(f"Row {row_index} parameters must be an object or JSON string.")

    def _has_meaningful_data(self, row: Mapping[str, object], parameters: Mapping[str, object]) -> bool:
        fields_to_check = [
            row.get("code"),
            row.get("name"),
            row.get("smiles"),
            row.get("canonical_smiles"),
            row.get("notes"),
            row.get("molblock"),
        ]
        return any(value not in (None, "") for value in fields_to_check) or bool(parameters)

    def _validate_sequence(self, records: Sequence[object]) -> None:
        for index, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise ValueError(f"JSON record {index} must be an object.")

    @staticmethod
    def _normalize_cell(value: object) -> object:
        if pd.isna(value):
            return None
        if isinstance(value, str):
            return value.strip()
        return value

    @staticmethod
    def _clean_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _parse_bool(value: object) -> bool:
        if value is None or value == "":
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)

        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")
