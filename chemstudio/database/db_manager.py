from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from chemstudio.database.models import MoleculeDetail, MoleculeImportRecord
from chemstudio.utils.file_utils import ensure_directory


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS molecules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    smiles TEXT,
    source TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS molecular_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    molecule_id INTEGER NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value REAL,
    FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS property_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    molecule_id INTEGER NOT NULL,
    property_name TEXT NOT NULL,
    property_value REAL,
    FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS formulas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula_name TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    composition_json TEXT NOT NULL,
    conditions_json TEXT NOT NULL DEFAULT '{}',
    predicted_property_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_molecular_features_molecule_id
ON molecular_features (molecule_id);

CREATE INDEX IF NOT EXISTS idx_property_data_molecule_id
ON property_data (molecule_id);

CREATE INDEX IF NOT EXISTS idx_molecular_features_feature_name
ON molecular_features (feature_name);

CREATE INDEX IF NOT EXISTS idx_property_data_property_name
ON property_data (property_name);
"""


class DatabaseManager:
    """Encapsulates SQLite schema management and CRUD operations."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        ensure_directory(self.db_path.parent)

    def connect(self) -> sqlite3.Connection:
        """Return a configured SQLite connection."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def initialize_database(self) -> None:
        """Create required tables and indexes."""
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)
            self._migrate_schema(connection)
            connection.commit()

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        """Apply lightweight schema migrations required by newer formula features."""
        self._ensure_column(connection, "formulas", "note", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "formulas", "conditions_json", "TEXT NOT NULL DEFAULT '{}'")

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in columns:
            return
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    def count_rows(self, table_name: str) -> int:
        """Return row count for a table."""
        with self.connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
            return int(row["count"]) if row is not None else 0

    def insert_molecule_record(self, record: MoleculeImportRecord) -> int:
        """Insert a single molecule plus feature/property rows."""
        with self.connect() as connection:
            molecule_id = self._insert_molecule_record(connection, record)
            connection.commit()
            return molecule_id

    def bulk_insert_records(self, records: list[MoleculeImportRecord]) -> list[int]:
        """Insert multiple molecule records in a transaction."""
        inserted_ids: list[int] = []
        with self.connect() as connection:
            for record in records:
                inserted_ids.append(self._insert_molecule_record(connection, record))
            connection.commit()
        return inserted_ids

    def _insert_molecule_record(self, connection: sqlite3.Connection, record: MoleculeImportRecord) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        cursor = connection.execute(
            """
            INSERT INTO molecules (name, smiles, source, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (record.name, record.smiles, record.source, created_at),
        )
        molecule_id = int(cursor.lastrowid)

        if record.features:
            connection.executemany(
                """
                INSERT INTO molecular_features (molecule_id, feature_name, feature_value)
                VALUES (?, ?, ?)
                """,
                [
                    (molecule_id, feature_name, float(feature_value))
                    for feature_name, feature_value in sorted(record.features.items())
                ],
            )

        if record.properties:
            connection.executemany(
                """
                INSERT INTO property_data (molecule_id, property_name, property_value)
                VALUES (?, ?, ?)
                """,
                [
                    (molecule_id, property_name, float(property_value))
                    for property_name, property_value in sorted(record.properties.items())
                ],
            )

        return molecule_id

    def list_molecules(self, search_text: str = "") -> list[dict[str, object]]:
        """Return lightweight molecule rows for selectors and tables."""
        query = """
            SELECT id, name, smiles, source, created_at
            FROM molecules
        """
        parameters: list[object] = []
        if search_text.strip():
            pattern = f"%{search_text.strip()}%"
            query += " WHERE name LIKE ? OR smiles LIKE ? OR source LIKE ?"
            parameters.extend([pattern, pattern, pattern])
        query += " ORDER BY id DESC"

        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def get_molecule_detail(self, molecule_id: int) -> MoleculeDetail | None:
        """Load a single molecule with pivoted feature and property dictionaries."""
        with self.connect() as connection:
            molecule_row = connection.execute(
                """
                SELECT id, name, smiles, source, created_at
                FROM molecules
                WHERE id = ?
                """,
                (molecule_id,),
            ).fetchone()
            if molecule_row is None:
                return None

            feature_rows = connection.execute(
                """
                SELECT feature_name, feature_value
                FROM molecular_features
                WHERE molecule_id = ?
                ORDER BY feature_name
                """,
                (molecule_id,),
            ).fetchall()
            property_rows = connection.execute(
                """
                SELECT property_name, property_value
                FROM property_data
                WHERE molecule_id = ?
                ORDER BY property_name
                """,
                (molecule_id,),
            ).fetchall()

        return MoleculeDetail(
            id=int(molecule_row["id"]),
            name=str(molecule_row["name"]),
            smiles=str(molecule_row["smiles"] or ""),
            source=str(molecule_row["source"] or ""),
            created_at=str(molecule_row["created_at"]),
            features={str(row["feature_name"]): float(row["feature_value"]) for row in feature_rows},
            properties={str(row["property_name"]): float(row["property_value"]) for row in property_rows},
        )

    def delete_molecule(self, molecule_id: int) -> bool:
        """Delete a molecule and all child records."""
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM molecules WHERE id = ?", (molecule_id,))
            connection.commit()
            return cursor.rowcount > 0

    def list_feature_names(self) -> list[str]:
        """Return all distinct feature names."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT feature_name FROM molecular_features ORDER BY feature_name"
            ).fetchall()
        return [str(row["feature_name"]) for row in rows]

    def list_property_names(self) -> list[str]:
        """Return all distinct property names."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT property_name FROM property_data ORDER BY property_name"
            ).fetchall()
        return [str(row["property_name"]) for row in rows]

    def get_wide_dataset(self, search_text: str = "") -> pd.DataFrame:
        """Return a wide dataframe joined from metadata, features, and properties."""
        molecules = pd.DataFrame(self.list_molecules(search_text=search_text))
        if molecules.empty:
            return pd.DataFrame(columns=["id", "name", "smiles", "source", "created_at"])

        with self.connect() as connection:
            feature_rows = pd.read_sql_query(
                "SELECT molecule_id, feature_name, feature_value FROM molecular_features",
                connection,
            )
            property_rows = pd.read_sql_query(
                "SELECT molecule_id, property_name, property_value FROM property_data",
                connection,
            )

        dataset = molecules.copy()

        if not feature_rows.empty:
            feature_frame = (
                feature_rows.pivot_table(
                    index="molecule_id",
                    columns="feature_name",
                    values="feature_value",
                    aggfunc="mean",
                )
                .reset_index()
                .rename(columns={"molecule_id": "id"})
            )
            dataset = dataset.merge(feature_frame, on="id", how="left")

        if not property_rows.empty:
            property_frame = (
                property_rows.pivot_table(
                    index="molecule_id",
                    columns="property_name",
                    values="property_value",
                    aggfunc="mean",
                )
                .reset_index()
                .rename(columns={"molecule_id": "id"})
            )
            overlap = [column for column in property_frame.columns if column in dataset.columns and column != "id"]
            if overlap:
                property_frame = property_frame.rename(columns={column: f"property__{column}" for column in overlap})
            dataset = dataset.merge(property_frame, on="id", how="left")

        ordered_columns = ["id", "name", "smiles", "source", "created_at"]
        remaining_columns = [column for column in dataset.columns if column not in ordered_columns]
        return dataset[ordered_columns + sorted(remaining_columns)]

    def save_formula(
        self,
        formula_name: str,
        composition: list[dict[str, object]],
        predicted_properties: dict[str, float],
        note: str = "",
        conditions: dict[str, float] | None = None,
    ) -> int:
        """Persist a predicted formula configuration."""
        created_at = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO formulas (formula_name, note, composition_json, conditions_json, predicted_property_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    formula_name,
                    note,
                    json.dumps(composition, ensure_ascii=False, indent=2),
                    json.dumps(conditions or {}, ensure_ascii=False, indent=2),
                    json.dumps(predicted_properties, ensure_ascii=False, indent=2),
                    created_at,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def save_formulation(
        self,
        formula_name: str,
        note: str,
        composition: list[dict[str, object]],
        target_values: dict[str, float],
        conditions: dict[str, float] | None = None,
    ) -> int:
        """Persist a formulation record used for formula learning and training."""
        return self.save_formula(
            formula_name=formula_name,
            note=note,
            composition=composition,
            conditions=conditions,
            predicted_properties=target_values,
        )

    def list_formulas(self, limit: int = 100) -> list[dict[str, object]]:
        """Return recently saved formulas."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, formula_name, note, composition_json, conditions_json, predicted_property_json, created_at
                FROM formulas
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_formulations(self, limit: int = 200) -> list[dict[str, object]]:
        """Return recently saved formulations for the formula-design module."""
        return self.list_formulas(limit=limit)

    def get_formulation(self, formulation_id: int) -> dict[str, object] | None:
        """Return one saved formulation row."""
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT id, formula_name, note, composition_json, conditions_json, predicted_property_json, created_at
                FROM formulas
                WHERE id = ?
                """,
                (formulation_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def delete_formulation(self, formulation_id: int) -> bool:
        """Delete a saved formulation."""
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM formulas WHERE id = ?", (formulation_id,))
            connection.commit()
            return cursor.rowcount > 0
