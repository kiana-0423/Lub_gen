from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.models import MoleculeImportRecord
from chemstudio.ml.featurizers import compute_mordred_descriptors
from chemstudio.utils.config import AppConfig
from chemstudio.utils.file_utils import normalize_field_name, read_tabular_file
from chemstudio.validation import validate_molecule_name, validate_smiles

try:  # pragma: no cover - dependency is declared, fallback keeps imports resilient
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover
    Parallel = None
    delayed = None

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, inchi, rdMolDescriptors
except ImportError:  # pragma: no cover
    Chem = None
    Descriptors = None
    inchi = None
    rdMolDescriptors = None


logger = logging.getLogger(__name__)


class DataImportService:
    """Loads tabular files, maps columns, and persists records into SQLite."""

    RESERVED_FIELDS = {
        "code",
        "name",
        "smiles",
        "canonical_smiles",
        "is_hidden",
        "notes",
        "molblock",
        "parameters",
        "source",
    }
    NAME_COLUMNS = {"name", "molecule_name", "material_name", "compound_name"}
    SMILES_COLUMNS = {"smiles", "structure", "structure_id", "canonical_smiles"}
    SOURCE_COLUMNS = {"source", "dataset", "origin"}
    PROPERTY_KEYWORDS = {
        "property",
        "performance",
        "target",
        "label",
        "conductivity",
        "viscosity",
        "stability",
        "capacity",
        "strength",
        "modulus",
        "retention",
    }

    def __init__(self, db_manager: DatabaseManager) -> None:
        """保存数据库访问依赖，供导入流程复用。"""
        self.db_manager = db_manager

    def import_file(self, file_path: str | Path) -> dict[str, object]:
        """Import a CSV, Excel, or JSON file into the database."""
        if Path(file_path).suffix.lower() == ".json":
            return self.import_json(file_path)
        dataframe = read_tabular_file(file_path)
        records = self.parse_dataframe(dataframe, source_label=Path(file_path).name, parameter_mode=True)
        inserted_ids = self.db_manager.bulk_insert_records(records)
        return {
            "file_path": str(file_path),
            "row_count": len(records),
            "inserted_ids": inserted_ids,
            "columns": list(dataframe.columns),
        }

    def import_json(self, file_path: str | Path) -> dict[str, object]:
        """Import JSON records into the database."""
        records = self.load_records(file_path)
        molecule_records = [self._record_to_import_record(record, source_label=Path(file_path).name) for record in records]
        inserted_ids = self.db_manager.bulk_insert_records(molecule_records)
        return {
            "file_path": str(file_path),
            "row_count": len(molecule_records),
            "inserted_ids": inserted_ids,
            "columns": sorted({key for record in records for key in record.keys()}),
        }

    def load_records(self, file_path: str | Path) -> list[dict[str, object]]:
        """Load CSV, Excel, or JSON records with import-service normalization."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            raw_records = self._load_json(path)
        elif suffix == ".csv":
            raw_records = self._frame_to_records(pd.read_csv(path))
        elif suffix in {".xlsx", ".xls"}:
            raw_records = self._frame_to_records(pd.read_excel(path))
        else:
            raise ValueError(f"Unsupported import format: {path.suffix or '<none>'}")

        records: list[dict[str, object]] = []
        for index, raw_record in enumerate(raw_records, start=1):
            normalized = self._normalize_record(raw_record, row_index=index)
            if normalized is not None:
                records.append(normalized)
        return records

    def parse_dataframe(
        self,
        dataframe: pd.DataFrame,
        source_label: str = "",
        *,
        parameter_mode: bool = False,
    ) -> list[MoleculeImportRecord]:
        """Convert a dataframe into molecule import records."""
        if dataframe.empty:
            return []

        mapped_columns = {column: normalize_field_name(column) for column in dataframe.columns}
        feature_columns: list[str] = []
        property_columns: list[str] = []
        parameter_columns: list[str] = []

        for original_name, normalized_name in mapped_columns.items():
            if normalized_name in self.NAME_COLUMNS | self.SMILES_COLUMNS | self.SOURCE_COLUMNS:
                continue
            if parameter_mode and str(original_name).startswith("parameter:"):
                parameter_columns.append(original_name)
                continue

            if normalized_name.startswith(("feature_", "feat_", "descriptor_")):
                feature_columns.append(original_name)
                continue
            if normalized_name.startswith(("property_", "prop_", "target_", "label_")):
                property_columns.append(original_name)
                continue

            series = pd.to_numeric(dataframe[original_name], errors="coerce")
            if series.notna().sum() == 0:
                continue

            if any(keyword in normalized_name for keyword in self.PROPERTY_KEYWORDS):
                property_columns.append(original_name)
            else:
                feature_columns.append(original_name)

        pending_records: list[dict[str, object]] = []
        for row_index, (_, row) in enumerate(dataframe.iterrows(), start=1):
            name = self._extract_text(row, mapped_columns, self.NAME_COLUMNS) or f"molecule_{row_index}"
            smiles = self._extract_text(row, mapped_columns, self.SMILES_COLUMNS)
            source = self._extract_text(row, mapped_columns, self.SOURCE_COLUMNS) or source_label

            features = self._collect_numeric_values(row, feature_columns)
            properties = self._collect_numeric_values(row, property_columns)
            parameters = self._collect_parameter_values(row, parameter_columns)

            if not any([name.strip(), smiles.strip(), features, properties]):
                continue
            standardized = self.validate_and_standardize(
                {
                    "name": name,
                    "smiles": smiles,
                    "source": source,
                    "parameters": parameters,
                }
            )

            pending_records.append(
                {
                    "standardized": standardized,
                    "source": source,
                    "features": features,
                    "properties": properties,
                }
            )

        descriptor_results = self._compute_descriptors_batch(
            [str(item["standardized"]["canonical_smiles"]) for item in pending_records]  # type: ignore[index]
        )

        records: list[MoleculeImportRecord] = []
        for item, descriptors in zip(pending_records, descriptor_results, strict=True):
            standardized = item["standardized"]
            if not isinstance(standardized, Mapping):
                raise ValueError("Internal import state is invalid.")
            records.append(
                MoleculeImportRecord(
                    name=str(standardized["name"]),
                    smiles=str(standardized["canonical_smiles"]),
                    source=str(item["source"]),
                    code=standardized.get("code") if isinstance(standardized.get("code"), str) else None,
                    input_smiles=str(standardized["input_smiles"]),
                    canonical_smiles=str(standardized["canonical_smiles"]),
                    inchi=str(standardized["inchi"]),
                    inchikey=str(standardized["inchikey"]),
                    molblock=str(standardized["molblock"]),
                    notes=str(standardized["notes"]),
                    is_hidden=bool(standardized["is_hidden"]),
                    parameters=dict(standardized["parameters"]),
                    descriptors=descriptors,
                    features=dict(item["features"]),
                    properties=dict(item["properties"]),
                )
            )

        return records

    def seed_mock_data_if_empty(self) -> bool:
        """Load bundled sample data on first startup."""
        if self.db_manager.count_rows("molecules") > 0:
            return False
        self.import_file(AppConfig.SAMPLE_DATA_PATH)
        return True

    def validate_and_standardize(self, payload: Mapping[str, object]) -> dict[str, object]:
        """Validate molecule input and add canonical structure metadata when RDKit is available."""
        smiles = str(payload.get("smiles") or payload.get("canonical_smiles") or "").strip()
        canonical_smiles = validate_smiles(smiles)

        if Chem is not None:
            molecule = Chem.MolFromSmiles(canonical_smiles)
            molecular_formula = rdMolDescriptors.CalcMolFormula(molecule) if rdMolDescriptors else ""
            molecular_weight = float(Descriptors.MolWt(molecule)) if Descriptors else 0.0
            inchi_value = inchi.MolToInchi(molecule) if inchi is not None else ""
            inchikey_value = inchi.MolToInchiKey(molecule) if inchi is not None else ""
        else:  # pragma: no cover
            molecular_formula = ""
            molecular_weight = 0.0
            inchi_value = ""
            inchikey_value = ""

        raw_parameters = payload.get("parameters") or {}
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("parameters must be a mapping.")
        parameters = {str(key): value for key, value in raw_parameters.items()}
        parameters.update(
            {
                "input_smiles": smiles,
                "molecular_formula": molecular_formula,
                "molecular_weight": round(molecular_weight, 6),
            }
        )

        return {
            "code": self._clean_text(payload.get("code")),
            "name": validate_molecule_name(self._clean_text(payload.get("name")) or canonical_smiles),
            "input_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "inchi": inchi_value,
            "inchikey": inchikey_value,
            "molblock": self._clean_text(payload.get("molblock")) or "",
            "notes": self._clean_text(payload.get("notes")) or "",
            "is_hidden": self._parse_bool(payload.get("is_hidden")),
            "parameters": parameters,
        }

    def _extract_text(self, row: pd.Series, mapped_columns: dict[str, str], aliases: set[str]) -> str:
        """按照别名集合从当前行里提取第一个非空文本值。"""
        for original_name, normalized_name in mapped_columns.items():
            if normalized_name not in aliases:
                continue
            value = row.get(original_name)
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _collect_numeric_values(self, row: pd.Series, columns: list[str]) -> dict[str, float]:
        """读取指定列中的数值，并清理统一的特征或属性前缀。"""
        values: dict[str, float] = {}
        for column in columns:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.isna(value):
                continue
            normalized_name = normalize_field_name(column)
            for prefix in ("feature_", "feat_", "descriptor_", "property_", "prop_", "target_", "label_"):
                if normalized_name.startswith(prefix):
                    normalized_name = normalized_name.removeprefix(prefix)
            values[normalized_name] = float(value)
        return values

    def _collect_parameter_values(self, row: pd.Series, columns: list[str]) -> dict[str, object]:
        values: dict[str, object] = {}
        for column in columns:
            value = self._normalize_cell(row.get(column))
            if value in (None, ""):
                continue
            parameter_key = str(column).split("parameter:", 1)[1]
            values[parameter_key] = value
        return values

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

    def _record_to_import_record(self, record: Mapping[str, object], *, source_label: str) -> MoleculeImportRecord:
        standardized = self.validate_and_standardize(record)
        return MoleculeImportRecord(
            name=str(standardized["name"]),
            smiles=str(standardized["canonical_smiles"]),
            source=source_label,
            code=standardized.get("code") if isinstance(standardized.get("code"), str) else None,
            input_smiles=str(standardized["input_smiles"]),
            canonical_smiles=str(standardized["canonical_smiles"]),
            inchi=str(standardized["inchi"]),
            inchikey=str(standardized["inchikey"]),
            molblock=str(standardized["molblock"]),
            notes=str(standardized["notes"]),
            is_hidden=bool(standardized["is_hidden"]),
            parameters=dict(standardized["parameters"]),
            descriptors=self._compute_import_descriptors(str(standardized["canonical_smiles"])),
        )

    def _compute_import_descriptors(self, smiles: str) -> dict[str, float]:
        """Generate Mordred descriptors during import."""
        return compute_mordred_descriptors(smiles)

    def _compute_descriptors_batch(self, smiles_list: list[str]) -> list[dict[str, float]]:
        """Generate Mordred descriptors for many SMILES strings with a safe parallel fallback."""
        if not smiles_list:
            return []
        if Parallel is None or delayed is None or len(smiles_list) < 50:
            return [self._compute_import_descriptors(smiles) for smiles in smiles_list]

        started_at = time.perf_counter()
        logger.info("Computing Mordred descriptors in parallel for %d molecules.", len(smiles_list))
        try:
            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(compute_mordred_descriptors)(smiles) for smiles in smiles_list
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning("Parallel Mordred descriptor calculation failed; falling back to serial: %s", exc)
            return [self._compute_import_descriptors(smiles) for smiles in smiles_list]

        elapsed_seconds = time.perf_counter() - started_at
        logger.info(
            "Computed Mordred descriptors for %d molecules in %.2f seconds.",
            len(smiles_list),
            elapsed_seconds,
        )
        return [dict(result) for result in results]

    def _compute_descriptors_parallel(self, smiles_list: list[str]) -> list[dict[str, float]]:
        """Backward-compatible wrapper for the batch Mordred descriptor path."""
        return self._compute_descriptors_batch(smiles_list)

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
        if isinstance(value, (Mapping, list, tuple)):
            return value
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
