from __future__ import annotations

from pathlib import Path

import pandas as pd

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.models import MoleculeImportRecord
from chemstudio.utils.config import AppConfig
from chemstudio.utils.file_utils import normalize_field_name, read_tabular_file


class DataImportService:
    """Loads tabular files, maps columns, and persists records into SQLite."""

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
        """Import a CSV or Excel file into the database."""
        dataframe = read_tabular_file(file_path)
        records = self.parse_dataframe(dataframe, source_label=Path(file_path).name)
        inserted_ids = self.db_manager.bulk_insert_records(records)
        return {
            "file_path": str(file_path),
            "row_count": len(records),
            "inserted_ids": inserted_ids,
            "columns": list(dataframe.columns),
        }

    def parse_dataframe(self, dataframe: pd.DataFrame, source_label: str = "") -> list[MoleculeImportRecord]:
        """Convert a dataframe into molecule import records."""
        if dataframe.empty:
            return []

        mapped_columns = {column: normalize_field_name(column) for column in dataframe.columns}
        feature_columns: list[str] = []
        property_columns: list[str] = []

        for original_name, normalized_name in mapped_columns.items():
            if normalized_name in self.NAME_COLUMNS | self.SMILES_COLUMNS | self.SOURCE_COLUMNS:
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

        records: list[MoleculeImportRecord] = []
        for row_index, (_, row) in enumerate(dataframe.iterrows(), start=1):
            name = self._extract_text(row, mapped_columns, self.NAME_COLUMNS) or f"molecule_{row_index}"
            smiles = self._extract_text(row, mapped_columns, self.SMILES_COLUMNS)
            source = self._extract_text(row, mapped_columns, self.SOURCE_COLUMNS) or source_label

            features = self._collect_numeric_values(row, feature_columns)
            properties = self._collect_numeric_values(row, property_columns)

            if not any([name.strip(), smiles.strip(), features, properties]):
                continue

            records.append(
                MoleculeImportRecord(
                    name=name,
                    smiles=smiles,
                    source=source,
                    features=features,
                    properties=properties,
                )
            )

        return records

    def seed_mock_data_if_empty(self) -> bool:
        """Load bundled sample data on first startup."""
        if self.db_manager.count_rows("molecules") > 0:
            return False
        self.import_file(AppConfig.SAMPLE_DATA_PATH)
        return True

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
