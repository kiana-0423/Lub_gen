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
    code TEXT UNIQUE,
    name TEXT NOT NULL,
    smiles TEXT,
    input_smiles TEXT NOT NULL DEFAULT '',
    canonical_smiles TEXT NOT NULL DEFAULT '',
    inchi TEXT NOT NULL DEFAULT '',
    inchikey TEXT NOT NULL DEFAULT '',
    molblock TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    is_hidden INTEGER NOT NULL DEFAULT 0,
    source TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS molecule_parameters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    molecule_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (molecule_id, key),
    FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS molecule_descriptors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    molecule_id INTEGER NOT NULL UNIQUE,
    descriptor_values_json TEXT NOT NULL DEFAULT '{}',
    fingerprint_bits TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE CASCADE
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

CREATE TABLE IF NOT EXISTS model_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    problem_type TEXT NOT NULL,
    target_name TEXT NOT NULL,
    feature_columns_json TEXT NOT NULL DEFAULT '[]',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    training_config_json TEXT NOT NULL DEFAULT '{}',
    artifact_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    molecule_id INTEGER,
    predicted_value REAL,
    predicted_label TEXT NOT NULL DEFAULT '',
    confidence REAL,
    input_features_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (model_id) REFERENCES model_records (id) ON DELETE CASCADE,
    FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_molecules_code
ON molecules (code);

CREATE INDEX IF NOT EXISTS idx_molecules_canonical_smiles
ON molecules (canonical_smiles);

CREATE INDEX IF NOT EXISTS idx_molecules_inchikey
ON molecules (inchikey);

CREATE INDEX IF NOT EXISTS idx_molecules_is_hidden
ON molecules (is_hidden);

CREATE INDEX IF NOT EXISTS idx_molecule_parameters_molecule_id
ON molecule_parameters (molecule_id);

CREATE INDEX IF NOT EXISTS idx_molecule_parameters_key
ON molecule_parameters (key);

CREATE INDEX IF NOT EXISTS idx_molecular_features_molecule_id
ON molecular_features (molecule_id);

CREATE INDEX IF NOT EXISTS idx_property_data_molecule_id
ON property_data (molecule_id);

CREATE INDEX IF NOT EXISTS idx_molecular_features_feature_name
ON molecular_features (feature_name);

CREATE INDEX IF NOT EXISTS idx_property_data_property_name
ON property_data (property_name);

CREATE INDEX IF NOT EXISTS idx_model_records_name
ON model_records (name);

CREATE INDEX IF NOT EXISTS idx_predictions_model_id
ON predictions (model_id);

CREATE INDEX IF NOT EXISTS idx_predictions_molecule_id
ON predictions (molecule_id);
"""


class DatabaseManager:
    """Encapsulates SQLite schema management and CRUD operations."""

    def __init__(self, db_path: Path | str) -> None:
        """保存数据库文件路径，并提前创建父目录。"""
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
        self._ensure_column(connection, "molecules", "code", "TEXT")
        self._ensure_column(connection, "molecules", "smiles", "TEXT")
        self._ensure_column(connection, "molecules", "source", "TEXT")
        self._ensure_column(connection, "molecules", "input_smiles", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "molecules", "canonical_smiles", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "molecules", "inchi", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "molecules", "inchikey", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "molecules", "molblock", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "molecules", "notes", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "molecules", "is_hidden", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column(connection, "molecules", "updated_at", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "formulas", "note", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "formulas", "conditions_json", "TEXT NOT NULL DEFAULT '{}'")
        connection.execute("UPDATE molecules SET updated_at = created_at WHERE updated_at = ''")
        connection.execute("UPDATE molecules SET canonical_smiles = smiles WHERE canonical_smiles = '' AND smiles IS NOT NULL")
        connection.execute("UPDATE molecules SET input_smiles = smiles WHERE input_smiles = '' AND smiles IS NOT NULL")
        connection.execute("UPDATE molecules SET smiles = canonical_smiles WHERE (smiles IS NULL OR smiles = '') AND canonical_smiles != ''")
        connection.execute("UPDATE molecules SET source = '' WHERE source IS NULL")

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        """在旧表缺失列时追加新列，避免重复迁移。"""
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
        """写入单个分子及其特征、属性明细行，并返回新 ID。"""
        created_at = datetime.now(timezone.utc).isoformat()
        canonical_smiles = record.canonical_smiles or record.smiles
        input_smiles = record.input_smiles or record.smiles
        cursor = connection.execute(
            """
            INSERT INTO molecules (
                code, name, smiles, input_smiles, canonical_smiles, inchi, inchikey,
                molblock, notes, is_hidden, source, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.code,
                record.name,
                canonical_smiles,
                input_smiles,
                canonical_smiles,
                record.inchi,
                record.inchikey,
                record.molblock,
                record.notes,
                int(record.is_hidden),
                record.source,
                created_at,
                created_at,
            ),
        )
        molecule_id = int(cursor.lastrowid)
        self._replace_parameters(connection, molecule_id, record.parameters, created_at)

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

        if record.descriptors:
            self._save_descriptors(connection, molecule_id, record.descriptors, created_at)

        return molecule_id

    def save_molecule(self, payload: dict[str, object], molecule_id: int | None = None) -> dict[str, object]:
        """Create or update a molecule and its free-form parameters."""
        now = datetime.now(timezone.utc).isoformat()
        target_id = molecule_id
        if target_id is None:
            target_id = self._find_existing_molecule_id(payload)

        name = str(payload.get("name") or payload.get("canonical_smiles") or payload.get("smiles") or "").strip()
        if not name:
            raise ValueError("Molecule name or SMILES is required.")

        canonical_smiles = str(payload.get("canonical_smiles") or payload.get("smiles") or "").strip()
        input_smiles = str(payload.get("input_smiles") or payload.get("smiles") or canonical_smiles).strip()
        smiles = canonical_smiles or input_smiles
        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a mapping.")

        with self.connect() as connection:
            if target_id is None:
                cursor = connection.execute(
                    """
                    INSERT INTO molecules (
                        code, name, smiles, input_smiles, canonical_smiles, inchi, inchikey,
                        molblock, notes, is_hidden, source, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._clean_nullable(payload.get("code")),
                        name,
                        smiles,
                        input_smiles,
                        canonical_smiles,
                        str(payload.get("inchi") or ""),
                        str(payload.get("inchikey") or ""),
                        str(payload.get("molblock") or ""),
                        str(payload.get("notes") or ""),
                        int(bool(payload.get("is_hidden", False))),
                        str(payload.get("source") or ""),
                        now,
                        now,
                    ),
                )
                target_id = int(cursor.lastrowid)
            else:
                existing = connection.execute("SELECT id FROM molecules WHERE id = ?", (target_id,)).fetchone()
                if existing is None:
                    raise ValueError(f"Molecule {target_id} does not exist.")
                connection.execute(
                    """
                    UPDATE molecules
                    SET code = ?, name = ?, smiles = ?, input_smiles = ?, canonical_smiles = ?,
                        inchi = ?, inchikey = ?, molblock = ?, notes = ?, is_hidden = ?,
                        source = COALESCE(NULLIF(?, ''), source), updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        self._clean_nullable(payload.get("code")),
                        name,
                        smiles,
                        input_smiles,
                        canonical_smiles,
                        str(payload.get("inchi") or ""),
                        str(payload.get("inchikey") or ""),
                        str(payload.get("molblock") or ""),
                        str(payload.get("notes") or ""),
                        int(bool(payload.get("is_hidden", False))),
                        str(payload.get("source") or ""),
                        now,
                        target_id,
                    ),
                )
            self._replace_parameters(connection, target_id, parameters, now)
            connection.commit()
        detail = self.get_molecule_detail(target_id)
        if detail is None:
            raise ValueError(f"Molecule {target_id} does not exist.")
        return self.serialize_molecule_detail(detail)

    def _find_existing_molecule_id(self, payload: dict[str, object]) -> int | None:
        code = self._clean_nullable(payload.get("code"))
        canonical_smiles = str(payload.get("canonical_smiles") or payload.get("smiles") or "").strip()
        if not code and not canonical_smiles:
            return None
        with self.connect() as connection:
            if code:
                row = connection.execute("SELECT id FROM molecules WHERE code = ?", (code,)).fetchone()
                if row is not None:
                    return int(row["id"])
            if canonical_smiles:
                row = connection.execute(
                    "SELECT id FROM molecules WHERE canonical_smiles = ?",
                    (canonical_smiles,),
                ).fetchone()
                if row is not None:
                    return int(row["id"])
        return None

    def list_molecules(
        self,
        search_text: str = "",
        *,
        keyword: str | None = None,
        include_hidden: bool = True,
        hidden_only: bool = False,
        parameter_filters: dict[str, object] | None = None,
        sort_by: str = "id",
        descending: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        """Return lightweight molecule rows for selectors and tables."""
        query = """
            SELECT id, code, name, smiles, input_smiles, canonical_smiles, inchi, inchikey,
                   is_hidden, source, created_at, updated_at
            FROM molecules
        """
        parameters: list[object] = []
        filters = self._build_molecule_filters(
            keyword=keyword or search_text,
            include_hidden=include_hidden,
            hidden_only=hidden_only,
            parameter_filters=parameter_filters,
            parameters=parameters,
        )
        if filters:
            query += " WHERE " + " AND ".join(filters)
        sort_column = self._resolve_molecule_sort_column(sort_by)
        direction = "DESC" if descending else "ASC"
        query += f" ORDER BY {sort_column} {direction}, id DESC"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            parameters.extend([int(limit), max(0, int(offset))])

        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def list_molecules_page(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        keyword: str | None = None,
        include_hidden: bool = False,
        hidden_only: bool = False,
        parameter_filters: dict[str, object] | None = None,
        sort_by: str = "updated_at",
        descending: bool = True,
    ) -> dict[str, object]:
        """Return paginated molecule rows with total count."""
        offset = max(page - 1, 0) * page_size
        items = self.list_molecules(
            keyword=keyword,
            include_hidden=include_hidden,
            hidden_only=hidden_only,
            parameter_filters=parameter_filters,
            sort_by=sort_by,
            descending=descending,
            limit=page_size,
            offset=offset,
        )
        total = self.count_molecules(
            keyword=keyword,
            include_hidden=include_hidden,
            hidden_only=hidden_only,
            parameter_filters=parameter_filters,
        )
        return {"page": page, "page_size": page_size, "total": total, "items": items}

    def count_molecules(
        self,
        *,
        keyword: str | None = None,
        include_hidden: bool = False,
        hidden_only: bool = False,
        parameter_filters: dict[str, object] | None = None,
    ) -> int:
        """Count molecules matching the same filters as paginated listings."""
        query = "SELECT COUNT(*) AS count FROM molecules"
        parameters: list[object] = []
        filters = self._build_molecule_filters(
            keyword=keyword,
            include_hidden=include_hidden,
            hidden_only=hidden_only,
            parameter_filters=parameter_filters,
            parameters=parameters,
        )
        if filters:
            query += " WHERE " + " AND ".join(filters)
        with self.connect() as connection:
            row = connection.execute(query, parameters).fetchone()
        return int(row["count"]) if row is not None else 0

    def get_molecule_detail(self, molecule_id: int) -> MoleculeDetail | None:
        """Load a single molecule with pivoted feature and property dictionaries."""
        with self.connect() as connection:
            molecule_row = connection.execute(
                """
                SELECT id, code, name, smiles, input_smiles, canonical_smiles, inchi, inchikey,
                       molblock, notes, is_hidden, source, created_at, updated_at
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
            parameter_rows = connection.execute(
                """
                SELECT key, value
                FROM molecule_parameters
                WHERE molecule_id = ?
                ORDER BY key
                """,
                (molecule_id,),
            ).fetchall()
            descriptor_row = connection.execute(
                """
                SELECT descriptor_values_json
                FROM molecule_descriptors
                WHERE molecule_id = ?
                """,
                (molecule_id,),
            ).fetchone()

        return MoleculeDetail(
            id=int(molecule_row["id"]),
            name=str(molecule_row["name"]),
            smiles=str(molecule_row["smiles"] or ""),
            source=str(molecule_row["source"] or ""),
            created_at=str(molecule_row["created_at"]),
            code=str(molecule_row["code"]) if molecule_row["code"] is not None else None,
            input_smiles=str(molecule_row["input_smiles"] or ""),
            canonical_smiles=str(molecule_row["canonical_smiles"] or molecule_row["smiles"] or ""),
            inchi=str(molecule_row["inchi"] or ""),
            inchikey=str(molecule_row["inchikey"] or ""),
            molblock=str(molecule_row["molblock"] or ""),
            notes=str(molecule_row["notes"] or ""),
            is_hidden=bool(molecule_row["is_hidden"]),
            updated_at=str(molecule_row["updated_at"] or molecule_row["created_at"]),
            parameters={str(row["key"]): str(row["value"]) for row in parameter_rows},
            descriptor_values=self._loads_json_dict(
                descriptor_row["descriptor_values_json"] if descriptor_row is not None else "{}"
            ),
            features={str(row["feature_name"]): float(row["feature_value"]) for row in feature_rows},
            properties={str(row["property_name"]): float(row["property_value"]) for row in property_rows},
        )

    def serialize_molecule_detail(self, detail: MoleculeDetail, *, include_detail: bool = True) -> dict[str, object]:
        """Convert a MoleculeDetail dataclass to the dictionary shape used by services/tests."""
        data: dict[str, object] = {
            "id": detail.id,
            "code": detail.code,
            "name": detail.name,
            "display_name": detail.name or detail.code or detail.canonical_smiles,
            "smiles": detail.smiles,
            "input_smiles": detail.input_smiles,
            "canonical_smiles": detail.canonical_smiles,
            "inchi": detail.inchi,
            "inchikey": detail.inchikey,
            "is_hidden": detail.is_hidden,
            "source": detail.source,
            "created_at": detail.created_at,
            "updated_at": detail.updated_at,
        }
        if include_detail:
            data.update(
                {
                    "molblock": detail.molblock,
                    "notes": detail.notes,
                    "parameters": detail.parameters,
                    "descriptor_values": detail.descriptor_values,
                    "features": detail.features,
                    "properties": detail.properties,
                }
            )
        return data

    def delete_molecule(self, molecule_id: int) -> bool:
        """Delete a molecule and all child records."""
        with self.connect() as connection:
            cursor = connection.execute("DELETE FROM molecules WHERE id = ?", (molecule_id,))
            connection.commit()
            return cursor.rowcount > 0

    def set_molecule_hidden_state(self, molecule_id: int, hidden: bool) -> dict[str, object] | None:
        """Update a molecule's hidden state and return the updated detail."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE molecules SET is_hidden = ?, updated_at = ? WHERE id = ?",
                (int(hidden), now, molecule_id),
            )
            connection.commit()
        if cursor.rowcount == 0:
            return None
        detail = self.get_molecule_detail(molecule_id)
        return self.serialize_molecule_detail(detail) if detail is not None else None

    def save_descriptors(
        self,
        molecule_id: int,
        descriptors: dict[str, object],
        fingerprint_bits: str = "",
    ) -> None:
        """Insert or replace descriptor JSON for a molecule."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM molecule_descriptors WHERE molecule_id = ?",
                (molecule_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO molecule_descriptors (
                        molecule_id, descriptor_values_json, fingerprint_bits, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (molecule_id, json.dumps(descriptors, ensure_ascii=False), fingerprint_bits, now, now),
                )
            else:
                connection.execute(
                    """
                    UPDATE molecule_descriptors
                    SET descriptor_values_json = ?, fingerprint_bits = ?, updated_at = ?
                    WHERE molecule_id = ?
                    """,
                    (json.dumps(descriptors, ensure_ascii=False), fingerprint_bits, now, molecule_id),
                )
            connection.commit()

    def _save_descriptors(
        self,
        connection: sqlite3.Connection,
        molecule_id: int,
        descriptors: dict[str, object],
        timestamp: str,
        fingerprint_bits: str = "",
    ) -> None:
        existing = connection.execute(
            "SELECT id FROM molecule_descriptors WHERE molecule_id = ?",
            (molecule_id,),
        ).fetchone()
        payload = json.dumps(descriptors, ensure_ascii=False)
        if existing is None:
            connection.execute(
                """
                INSERT INTO molecule_descriptors (
                    molecule_id, descriptor_values_json, fingerprint_bits, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (molecule_id, payload, fingerprint_bits, timestamp, timestamp),
            )
        else:
            connection.execute(
                """
                UPDATE molecule_descriptors
                SET descriptor_values_json = ?, fingerprint_bits = ?, updated_at = ?
                WHERE molecule_id = ?
                """,
                (payload, fingerprint_bits, timestamp, molecule_id),
            )

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

    def save_model_record(
        self,
        *,
        name: str,
        model_type: str,
        problem_type: str,
        target_name: str,
        feature_columns: list[str],
        metrics: dict[str, object],
        training_config: dict[str, object] | None = None,
        artifact_path: str = "",
    ) -> int:
        """Persist trained-model metadata."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO model_records (
                    name, model_type, problem_type, target_name, feature_columns_json,
                    metrics_json, training_config_json, artifact_path, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    model_type,
                    problem_type,
                    target_name,
                    json.dumps(feature_columns, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                    json.dumps(training_config or {}, ensure_ascii=False),
                    artifact_path,
                    now,
                    now,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_model_records(self, limit: int = 100) -> list[dict[str, object]]:
        """Return saved model metadata rows."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, model_type, problem_type, target_name, feature_columns_json,
                       metrics_json, training_config_json, artifact_path, created_at, updated_at
                FROM model_records
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_prediction_record(
        self,
        *,
        model_id: int,
        molecule_id: int | None = None,
        predicted_value: float | None = None,
        predicted_label: str = "",
        confidence: float | None = None,
        input_features: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> int:
        """Persist a prediction metadata row."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    model_id, molecule_id, predicted_value, predicted_label, confidence,
                    input_features_json, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    molecule_id,
                    predicted_value,
                    predicted_label,
                    confidence,
                    json.dumps(input_features or {}, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_wide_dataset(self, search_text: str = "", *, include_mordred: bool = False) -> pd.DataFrame:
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
            descriptor_rows = pd.read_sql_query(
                "SELECT molecule_id, descriptor_values_json FROM molecule_descriptors",
                connection,
            ) if include_mordred else pd.DataFrame()

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

        if include_mordred and not descriptor_rows.empty:
            descriptor_parts: list[dict[str, object]] = []
            for _, row in descriptor_rows.iterrows():
                descriptor_values = self._loads_json_dict(row["descriptor_values_json"])
                if not descriptor_values:
                    continue
                descriptor_parts.append({"id": int(row["molecule_id"]), **descriptor_values})
            if descriptor_parts:
                descriptor_frame = pd.DataFrame(descriptor_parts)
                overlap = [column for column in descriptor_frame.columns if column in dataset.columns and column != "id"]
                if overlap:
                    descriptor_frame = descriptor_frame.rename(
                        columns={column: f"descriptor__{column}" for column in overlap}
                    )
                dataset = dataset.merge(descriptor_frame, on="id", how="left")

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

    def _replace_parameters(
        self,
        connection: sqlite3.Connection,
        molecule_id: int,
        parameters: dict[str, object],
        timestamp: str,
    ) -> None:
        """Replace a molecule's key-value parameters in one transaction."""
        connection.execute("DELETE FROM molecule_parameters WHERE molecule_id = ?", (molecule_id,))
        if not parameters:
            return
        connection.executemany(
            """
            INSERT INTO molecule_parameters (molecule_id, key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (molecule_id, str(key), "" if value is None else str(value), timestamp, timestamp)
                for key, value in sorted(parameters.items())
            ],
        )

    def _build_molecule_filters(
        self,
        *,
        keyword: str | None,
        include_hidden: bool,
        hidden_only: bool,
        parameter_filters: dict[str, object] | None,
        parameters: list[object],
    ) -> list[str]:
        """Build SQL snippets and append bound parameters for molecule filtering."""
        filters: list[str] = []
        if hidden_only:
            filters.append("is_hidden = 1")
        elif not include_hidden:
            filters.append("is_hidden = 0")

        if keyword and keyword.strip():
            pattern = f"%{keyword.strip()}%"
            filters.append(
                """
                (
                    name LIKE ? OR code LIKE ? OR smiles LIKE ? OR input_smiles LIKE ?
                    OR canonical_smiles LIKE ? OR inchikey LIKE ? OR source LIKE ?
                    OR EXISTS (
                        SELECT 1 FROM molecule_parameters
                        WHERE molecule_parameters.molecule_id = molecules.id
                          AND (key LIKE ? OR value LIKE ?)
                    )
                )
                """
            )
            parameters.extend([pattern, pattern, pattern, pattern, pattern, pattern, pattern, pattern, pattern])

        for key, value in (parameter_filters or {}).items():
            filters.append(
                """
                EXISTS (
                    SELECT 1 FROM molecule_parameters
                    WHERE molecule_parameters.molecule_id = molecules.id
                      AND key = ? AND value = ?
                )
                """
            )
            parameters.extend([str(key), str(value)])

        return filters

    def _resolve_molecule_sort_column(self, sort_by: str) -> str:
        """Resolve public sort names to safe SQL column names."""
        allowed = {
            "id": "id",
            "name": "name",
            "code": "code",
            "smiles": "smiles",
            "canonical_smiles": "canonical_smiles",
            "created_at": "created_at",
            "updated_at": "updated_at",
            "source": "source",
        }
        return allowed.get(sort_by, "id")

    def _loads_json_dict(self, raw_value: object) -> dict[str, object]:
        """Load a JSON object, returning an empty dict on invalid legacy values."""
        try:
            parsed = json.loads(str(raw_value or "{}"))
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    @staticmethod
    def _clean_nullable(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
