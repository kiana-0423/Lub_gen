from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from chemstudio.constants import LUBRICANT_PROPERTY_UNITS
from chemstudio.database.models import MoleculeDetail, MoleculeImportRecord
from chemstudio.utils.file_utils import ensure_directory


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS molecules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE,
    name TEXT NOT NULL,
    material_type_id INTEGER DEFAULT NULL,
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
    updated_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (material_type_id) REFERENCES material_types (id) ON DELETE SET NULL
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

CREATE TABLE IF NOT EXISTS material_types (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name TEXT NOT NULL,
    category TEXT NOT NULL,
    sub_category TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    typical_application TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (type_name, category, sub_category)
);

CREATE TABLE IF NOT EXISTS lubricant_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    molecule_id INTEGER NOT NULL,
    property_name TEXT NOT NULL,
    property_value REAL NOT NULL,
    property_unit TEXT NOT NULL DEFAULT '',
    test_standard TEXT NOT NULL DEFAULT '',
    test_condition_json TEXT NOT NULL DEFAULT '{}',
    is_blend_property INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (molecule_id, property_name),
    FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lubprop_molecule ON lubricant_properties(molecule_id);
CREATE INDEX IF NOT EXISTS idx_lubprop_name ON lubricant_properties(property_name);

CREATE TABLE IF NOT EXISTS formula_components (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula_id INTEGER NOT NULL,
    molecule_id INTEGER NOT NULL,
    component_role TEXT NOT NULL DEFAULT 'additive',
    ratio REAL NOT NULL DEFAULT 0.0,
    concentration REAL,
    concentration_unit TEXT NOT NULL DEFAULT 'wt%',
    sort_order INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (formula_id) REFERENCES formulas (id) ON DELETE CASCADE,
    FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_fc_formula ON formula_components(formula_id);
CREATE INDEX IF NOT EXISTS idx_fc_molecule ON formula_components(molecule_id);

CREATE TABLE IF NOT EXISTS additive_compatibilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    additive_id INTEGER NOT NULL,
    base_oil_id INTEGER NOT NULL,
    compatibility_score REAL,
    solubility TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (additive_id, base_oil_id),
    FOREIGN KEY (additive_id) REFERENCES molecules (id) ON DELETE CASCADE,
    FOREIGN KEY (base_oil_id) REFERENCES molecules (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS formula_test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula_id INTEGER NOT NULL,
    test_name TEXT NOT NULL,
    test_standard TEXT NOT NULL DEFAULT '',
    test_condition_json TEXT NOT NULL DEFAULT '{}',
    result_value REAL,
    result_unit TEXT NOT NULL DEFAULT '',
    is_predicted INTEGER NOT NULL DEFAULT 0,
    model_id INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (formula_id) REFERENCES formulas (id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES model_records (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_ftr_formula ON formula_test_results(formula_id);
"""

SCHEMA_TABLE_NAMES = {
    "molecules",
    "molecule_parameters",
    "molecule_descriptors",
    "molecular_features",
    "property_data",
    "formulas",
    "model_records",
    "predictions",
    "material_types",
    "lubricant_properties",
    "formula_components",
    "additive_compatibilities",
    "formula_test_results",
}

MIGRATED_COLUMNS = {
    "molecules": {
        "code": "TEXT",
        "smiles": "TEXT",
        "source": "TEXT",
        "input_smiles": "TEXT NOT NULL DEFAULT ''",
        "canonical_smiles": "TEXT NOT NULL DEFAULT ''",
        "inchi": "TEXT NOT NULL DEFAULT ''",
        "inchikey": "TEXT NOT NULL DEFAULT ''",
        "molblock": "TEXT NOT NULL DEFAULT ''",
        "notes": "TEXT NOT NULL DEFAULT ''",
        "is_hidden": "INTEGER NOT NULL DEFAULT 0",
        "material_type_id": "INTEGER DEFAULT NULL",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    },
    "formulas": {
        "note": "TEXT NOT NULL DEFAULT ''",
        "conditions_json": "TEXT NOT NULL DEFAULT '{}'",
    },
    "molecule_descriptors": {
        "descriptor_values_json": "TEXT NOT NULL DEFAULT '{}'",
        "fingerprint_bits": "TEXT NOT NULL DEFAULT ''",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    },
}

ALLOWED_SORT_DIRECTIONS = frozenset({"ASC", "DESC"})


class ClosingConnection(sqlite3.Connection):
    """SQLite connection that closes when used as a context manager."""

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        result = super().__exit__(exc_type, exc_value, traceback)
        self.close()
        return bool(result)


class DatabaseManager:
    """Encapsulates SQLite schema management and CRUD operations."""

    def __init__(self, db_path: Path | str) -> None:
        """保存数据库文件路径，并提前创建父目录。"""
        self.db_path = Path(db_path)
        ensure_directory(self.db_path.parent)

    def connect(self) -> sqlite3.Connection:
        """Return a configured SQLite connection."""
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
            factory=ClosingConnection,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.execute("PRAGMA busy_timeout = 30000;")
        connection.execute("PRAGMA journal_mode = WAL;")
        connection.execute("PRAGMA synchronous = NORMAL;")
        return connection

    def initialize_database(self) -> None:
        """Create required tables and indexes."""
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)
            self._migrate_schema(connection)
            self._seed_material_types(connection)
            self.migrate_property_to_lubricant(connection)
            connection.commit()

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        """Apply lightweight schema migrations required by newer formula features."""
        self._drop_unique_canonical_smiles_constraint_if_needed(connection)
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
        self._ensure_column(connection, "molecules", "material_type_id", "INTEGER DEFAULT NULL")
        self._ensure_column(connection, "molecules", "updated_at", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "formulas", "note", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "formulas", "conditions_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "molecule_descriptors", "descriptor_values_json", "TEXT NOT NULL DEFAULT '{}'")
        self._ensure_column(connection, "molecule_descriptors", "fingerprint_bits", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "molecule_descriptors", "created_at", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(connection, "molecule_descriptors", "updated_at", "TEXT NOT NULL DEFAULT ''")
        connection.execute("UPDATE molecules SET updated_at = created_at WHERE updated_at = ''")
        connection.execute("UPDATE molecules SET canonical_smiles = smiles WHERE canonical_smiles = '' AND smiles IS NOT NULL")
        connection.execute("UPDATE molecules SET input_smiles = smiles WHERE input_smiles = '' AND smiles IS NOT NULL")
        connection.execute("UPDATE molecules SET smiles = canonical_smiles WHERE (smiles IS NULL OR smiles = '') AND canonical_smiles != ''")
        connection.execute("UPDATE molecules SET source = '' WHERE source IS NULL")
        descriptor_columns = self._list_columns(connection, "molecule_descriptors")
        if "descriptor_values" in descriptor_columns:
            connection.execute(
                """
                UPDATE molecule_descriptors
                SET descriptor_values_json = descriptor_values
                WHERE descriptor_values_json = '{}'
                  AND descriptor_values IS NOT NULL
                  AND descriptor_values != ''
                """
            )
        connection.execute("UPDATE molecule_descriptors SET updated_at = created_at WHERE updated_at = ''")

    def _drop_unique_canonical_smiles_constraint_if_needed(self, connection: sqlite3.Connection) -> None:
        """Allow multiple experimental samples to share the same molecule structure."""
        unique_indexes = connection.execute("PRAGMA index_list(molecules)").fetchall()
        for index_row in unique_indexes:
            if int(index_row["unique"]) != 1:
                continue
            index_name = str(index_row["name"])
            quoted_index_name = '"' + index_name.replace('"', '""') + '"'
            index_columns = [
                str(column_row["name"])
                for column_row in connection.execute(f"PRAGMA index_info({quoted_index_name})").fetchall()
            ]
            if index_columns != ["canonical_smiles"]:
                continue

            if not index_name.startswith("sqlite_autoindex_"):
                connection.execute(f"DROP INDEX IF EXISTS {quoted_index_name}")
                return

            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("PRAGMA legacy_alter_table = ON")
            connection.execute("ALTER TABLE molecules RENAME TO molecules_legacy")
            connection.execute(
                """
                CREATE TABLE molecules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT UNIQUE,
                    name TEXT NOT NULL,
                    material_type_id INTEGER DEFAULT NULL,
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
                    updated_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (material_type_id) REFERENCES material_types (id) ON DELETE SET NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO molecules (
                    id, code, name, material_type_id, smiles, input_smiles, canonical_smiles, inchi, inchikey,
                    molblock, notes, is_hidden, source, created_at, updated_at
                )
                SELECT
                    id, code, name, NULL, smiles, input_smiles, canonical_smiles, inchi, inchikey,
                    molblock, notes, is_hidden, source, created_at, updated_at
                FROM molecules_legacy
                """
            )
            connection.execute("DROP TABLE molecules_legacy")
            connection.execute("PRAGMA legacy_alter_table = OFF")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_molecules_code ON molecules (code)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_molecules_canonical_smiles ON molecules (canonical_smiles)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_molecules_inchikey ON molecules (inchikey)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_molecules_is_hidden ON molecules (is_hidden)")
            return

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        """在旧表缺失列时追加新列，避免重复迁移。"""
        table_identifier = self._quote_identifier(table_name, SCHEMA_TABLE_NAMES)
        allowed_columns = MIGRATED_COLUMNS.get(table_name, {})
        if allowed_columns.get(column_name) != column_definition:
            raise ValueError(f"Unsupported schema migration column: {table_name}.{column_name}")
        column_identifier = self._quote_identifier(column_name, set(allowed_columns))
        columns = self._list_columns(connection, table_name)
        if column_name in columns:
            return
        connection.execute(f"ALTER TABLE {table_identifier} ADD COLUMN {column_identifier} {column_definition}")

    def _list_columns(self, connection: sqlite3.Connection, table_name: str) -> set[str]:
        """Return SQLite column names for an allowlisted table."""
        table_identifier = self._quote_identifier(table_name, SCHEMA_TABLE_NAMES)
        return {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table_identifier})").fetchall()
        }

    def count_rows(self, table_name: str) -> int:
        """Return row count for a table."""
        table_identifier = self._quote_identifier(table_name, SCHEMA_TABLE_NAMES)
        with self.connect() as connection:
            row = connection.execute(f"SELECT COUNT(*) AS count FROM {table_identifier}").fetchone()
            return int(row["count"]) if row is not None else 0

    def _seed_material_types(self, connection: sqlite3.Connection) -> None:
        """Insert built-in lubricant material categories once."""
        count_row = connection.execute("SELECT COUNT(*) AS count FROM material_types").fetchone()
        if count_row is not None and int(count_row["count"]) > 0:
            return
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            ("base_oil", "mineral_oil", "paraffinic", "石蜡基矿物油"),
            ("base_oil", "mineral_oil", "naphthenic", "环烷基矿物油"),
            ("base_oil", "synthetic", "PAO", "聚α-烯烃"),
            ("base_oil", "synthetic", "ester", "合成酯"),
            ("base_oil", "synthetic", "PAG", "聚醚/聚烷撑二醇"),
            ("base_oil", "synthetic", "silicone", "硅油"),
            ("base_oil", "vegetable", "natural_ester", "天然酯/植物油"),
            ("additive", "antioxidant", "phenolic", "酚类抗氧剂"),
            ("additive", "antioxidant", "aminic", "胺类抗氧剂"),
            ("additive", "antioxidant", "ZDDP", "二烷基二硫代磷酸锌"),
            ("additive", "antiwear", "phosphorus", "含磷抗磨剂"),
            ("additive", "antiwear", "sulfur", "含硫抗磨剂"),
            ("additive", "extreme_pressure", "sulfur_phosphorus", "硫磷型极压剂"),
            ("additive", "extreme_pressure", "borate", "硼酸盐极压剂"),
            ("additive", "friction_modifier", "organic", "有机摩擦改进剂"),
            ("additive", "friction_modifier", "MoDTC", "二硫代氨基甲酸钼"),
            ("additive", "viscosity_index_improver", "PMA", "聚甲基丙烯酸酯"),
            ("additive", "viscosity_index_improver", "OCP", "烯烃共聚物"),
            ("additive", "corrosion_inhibitor", "organic", "有机缓蚀剂"),
            ("additive", "detergent", "sulfonate", "磺酸盐清净剂"),
            ("additive", "dispersant", "succinimide", "丁二酰亚胺分散剂"),
            ("additive", "pour_point_depressant", "PMA", "聚甲基丙烯酸酯降凝剂"),
            ("additive", "defoamer", "silicone", "硅油抗泡剂"),
        ]
        connection.executemany(
            """
            INSERT OR IGNORE INTO material_types (
                type_name, category, sub_category, description, typical_application, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, '', ?, ?)
            """,
            [(type_name, category, sub_category, description, now, now) for type_name, category, sub_category, description in rows],
        )

    def seed_material_types(self) -> None:
        """Public entry point for reseeding built-in lubricant material categories."""
        with self.connect() as connection:
            self._seed_material_types(connection)
            connection.commit()

    def list_material_types(self, type_name: str | None = None) -> list[dict[str, object]]:
        """Return built-in lubricant material categories."""
        query = """
            SELECT id, type_name, category, sub_category, description, typical_application, created_at, updated_at
            FROM material_types
        """
        parameters: list[object] = []
        if type_name is not None:
            query += " WHERE type_name = ?"
            parameters.append(type_name)
        query += " ORDER BY type_name, category, sub_category"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def migrate_property_to_lubricant(self, connection: sqlite3.Connection | None = None) -> None:
        """Copy known lubricant properties from legacy property_data into rich property rows."""
        property_names = set(LUBRICANT_PROPERTY_UNITS)
        if not property_names:
            return

        def migrate(active_connection: sqlite3.Connection) -> None:
            placeholders = ", ".join("?" for _ in property_names)
            rows = active_connection.execute(
                f"""
                SELECT molecule_id, property_name, property_value
                FROM property_data
                WHERE property_name IN ({placeholders})
                """,
                sorted(property_names),
            ).fetchall()
            if not rows:
                return
            now = datetime.now(timezone.utc).isoformat()
            active_connection.executemany(
                """
                INSERT INTO lubricant_properties (
                    molecule_id, property_name, property_value, property_unit,
                    test_standard, test_condition_json, is_blend_property, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, '', '{}', 0, ?, ?)
                ON CONFLICT(molecule_id, property_name) DO NOTHING
                """,
                [
                    (
                        int(row["molecule_id"]),
                        str(row["property_name"]),
                        float(row["property_value"]),
                        LUBRICANT_PROPERTY_UNITS.get(str(row["property_name"]), ""),
                        now,
                        now,
                    )
                    for row in rows
                ],
            )

        if connection is not None:
            migrate(connection)
            return
        with self.connect() as owned_connection:
            migrate(owned_connection)
            owned_connection.commit()

    def save_lubricant_property(
        self,
        molecule_id: int,
        property_name: str,
        property_value: float,
        property_unit: str = "",
        test_standard: str = "",
        test_condition: dict[str, object] | None = None,
        is_blend_property: bool = False,
    ) -> int:
        """Upsert one lubricant property for a molecule."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO lubricant_properties (
                    molecule_id, property_name, property_value, property_unit, test_standard,
                    test_condition_json, is_blend_property, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(molecule_id, property_name) DO UPDATE SET
                    property_value = excluded.property_value,
                    property_unit = excluded.property_unit,
                    test_standard = excluded.test_standard,
                    test_condition_json = excluded.test_condition_json,
                    is_blend_property = excluded.is_blend_property,
                    updated_at = excluded.updated_at
                """,
                (
                    molecule_id,
                    property_name,
                    float(property_value),
                    property_unit,
                    test_standard,
                    json.dumps(test_condition or {}, ensure_ascii=False),
                    int(is_blend_property),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT id FROM lubricant_properties WHERE molecule_id = ? AND property_name = ?",
                (molecule_id, property_name),
            ).fetchone()
            connection.commit()
        return int(row["id"]) if row is not None else int(cursor.lastrowid)

    def save_formula_components(self, formula_id: int, components: list[dict[str, object]]) -> None:
        """Replace components for a formula."""
        with self.connect() as connection:
            self._replace_formula_components(connection, formula_id, components)
            connection.commit()

    def _replace_formula_components(
        self,
        connection: sqlite3.Connection,
        formula_id: int,
        components: list[dict[str, object]],
    ) -> None:
        connection.execute("DELETE FROM formula_components WHERE formula_id = ?", (formula_id,))
        if not components:
            return
        connection.executemany(
            """
            INSERT INTO formula_components (
                formula_id, molecule_id, component_role, ratio, concentration,
                concentration_unit, sort_order, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    formula_id,
                    int(component["molecule_id"]),
                    str(component.get("component_role") or "additive"),
                    float(component.get("ratio") or 0.0),
                    None if component.get("concentration") is None else float(component["concentration"]),
                    str(component.get("concentration_unit") or "wt%"),
                    int(component.get("sort_order") or index),
                    str(component.get("notes") or ""),
                )
                for index, component in enumerate(components)
                if component.get("molecule_id") is not None
            ],
        )

    def get_formula_components(self, formula_id: int) -> list[dict[str, object]]:
        """Return formula components with molecule labels."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT fc.id, fc.formula_id, fc.molecule_id, fc.component_role, fc.ratio,
                       fc.concentration, fc.concentration_unit, fc.sort_order, fc.notes,
                       m.name, m.smiles, m.material_type_id,
                       mt.type_name AS material_type_name,
                       mt.category AS material_category,
                       mt.sub_category AS material_sub_category
                FROM formula_components AS fc
                JOIN molecules AS m ON m.id = fc.molecule_id
                LEFT JOIN material_types AS mt ON mt.id = m.material_type_id
                WHERE fc.formula_id = ?
                ORDER BY fc.sort_order, fc.id
                """,
                (formula_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_additive_compatibility(
        self,
        additive_id: int,
        base_oil_id: int,
        compatibility_score: float | None = None,
        solubility: str = "",
        notes: str = "",
    ) -> int:
        """Upsert additive/base-oil compatibility data."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO additive_compatibilities (
                    additive_id, base_oil_id, compatibility_score, solubility, notes, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(additive_id, base_oil_id) DO UPDATE SET
                    compatibility_score = excluded.compatibility_score,
                    solubility = excluded.solubility,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (additive_id, base_oil_id, compatibility_score, solubility, notes, now, now),
            )
            row = connection.execute(
                "SELECT id FROM additive_compatibilities WHERE additive_id = ? AND base_oil_id = ?",
                (additive_id, base_oil_id),
            ).fetchone()
            connection.commit()
        return int(row["id"]) if row is not None else int(cursor.lastrowid)

    def get_additive_compatibilities(
        self,
        additive_id: int | None = None,
        base_oil_id: int | None = None,
    ) -> list[dict[str, object]]:
        """Return additive compatibility rows with molecule names."""
        query = """
            SELECT ac.id, ac.additive_id, additive.name AS additive_name,
                   ac.base_oil_id, base_oil.name AS base_oil_name,
                   ac.compatibility_score, ac.solubility, ac.notes, ac.created_at, ac.updated_at
            FROM additive_compatibilities AS ac
            JOIN molecules AS additive ON additive.id = ac.additive_id
            JOIN molecules AS base_oil ON base_oil.id = ac.base_oil_id
        """
        filters: list[str] = []
        parameters: list[object] = []
        if additive_id is not None:
            filters.append("ac.additive_id = ?")
            parameters.append(additive_id)
        if base_oil_id is not None:
            filters.append("ac.base_oil_id = ?")
            parameters.append(base_oil_id)
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY ac.id DESC"
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def save_formula_test_result(
        self,
        formula_id: int,
        test_name: str,
        result_value: float | None,
        test_standard: str = "",
        test_condition: dict[str, object] | None = None,
        result_unit: str = "",
        is_predicted: bool = False,
        model_id: int | None = None,
    ) -> int:
        """Insert a test result for a saved formula."""
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO formula_test_results (
                    formula_id, test_name, test_standard, test_condition_json,
                    result_value, result_unit, is_predicted, model_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    formula_id,
                    test_name,
                    test_standard,
                    json.dumps(test_condition or {}, ensure_ascii=False),
                    result_value,
                    result_unit,
                    int(is_predicted),
                    model_id,
                    now,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_formula_test_results(self, formula_id: int) -> list[dict[str, object]]:
        """Return test results for a formula."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, formula_id, test_name, test_standard, test_condition_json,
                       result_value, result_unit, is_predicted, model_id, created_at
                FROM formula_test_results
                WHERE formula_id = ?
                ORDER BY id DESC
                """,
                (formula_id,),
            ).fetchall()
        return [dict(row) for row in rows]

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
        """写入或更新单个分子及其特征、属性明细行，并返回目标 ID。"""
        created_at = datetime.now(timezone.utc).isoformat()
        canonical_smiles = record.canonical_smiles or record.smiles
        input_smiles = record.input_smiles or record.smiles
        molecule_id = self._find_existing_molecule_id_for_record(connection, record.code, canonical_smiles)
        if molecule_id is None:
            cursor = connection.execute(
                """
                INSERT INTO molecules (
                    code, name, material_type_id, smiles, input_smiles, canonical_smiles, inchi, inchikey,
                    molblock, notes, is_hidden, source, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.code,
                    record.name,
                    record.material_type_id,
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
        else:
            connection.execute(
                """
                UPDATE molecules
                SET code = ?, name = ?, material_type_id = ?, smiles = ?, input_smiles = ?, canonical_smiles = ?,
                    inchi = ?, inchikey = ?, molblock = ?, notes = ?, is_hidden = ?,
                    source = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    record.code,
                    record.name,
                    record.material_type_id,
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
                    molecule_id,
                ),
            )
        self._replace_parameters(connection, molecule_id, record.parameters, created_at)
        self._replace_feature_rows(connection, molecule_id, record.features)
        self._replace_property_rows(connection, molecule_id, record.properties)

        if record.descriptors:
            self._save_descriptors(connection, molecule_id, record.descriptors, created_at)

        return molecule_id

    def _find_existing_molecule_id_for_record(
        self,
        connection: sqlite3.Connection,
        code: str | None,
        canonical_smiles: str,
    ) -> int | None:
        if code:
            row = connection.execute("SELECT id FROM molecules WHERE code = ?", (code,)).fetchone()
            if row is not None:
                return int(row["id"])
            return None
        if canonical_smiles:
            row = connection.execute(
                "SELECT id FROM molecules WHERE canonical_smiles = ?",
                (canonical_smiles,),
            ).fetchone()
            if row is not None:
                return int(row["id"])
        return None

    def _replace_feature_rows(
        self,
        connection: sqlite3.Connection,
        molecule_id: int,
        features: dict[str, float],
    ) -> None:
        connection.execute("DELETE FROM molecular_features WHERE molecule_id = ?", (molecule_id,))
        if features:
            connection.executemany(
                """
                INSERT INTO molecular_features (molecule_id, feature_name, feature_value)
                VALUES (?, ?, ?)
                """,
                [
                    (molecule_id, feature_name, float(feature_value))
                    for feature_name, feature_value in sorted(features.items())
                ],
            )

    def _replace_property_rows(
        self,
        connection: sqlite3.Connection,
        molecule_id: int,
        properties: dict[str, float],
    ) -> None:
        connection.execute("DELETE FROM property_data WHERE molecule_id = ?", (molecule_id,))
        if properties:
            connection.executemany(
                """
                INSERT INTO property_data (molecule_id, property_name, property_value)
                VALUES (?, ?, ?)
                """,
                [
                    (molecule_id, property_name, float(property_value))
                    for property_name, property_value in sorted(properties.items())
                ],
            )

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
                        code, name, material_type_id, smiles, input_smiles, canonical_smiles, inchi, inchikey,
                        molblock, notes, is_hidden, source, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._clean_nullable(payload.get("code")),
                        name,
                        self._clean_int(payload.get("material_type_id")),
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
                    SET code = ?, name = ?, material_type_id = ?, smiles = ?, input_smiles = ?, canonical_smiles = ?,
                        inchi = ?, inchikey = ?, molblock = ?, notes = ?, is_hidden = ?,
                        source = COALESCE(NULLIF(?, ''), source), updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        self._clean_nullable(payload.get("code")),
                        name,
                        self._clean_int(payload.get("material_type_id")),
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
        material_type_id: int | None = None,
        sort_by: str = "id",
        descending: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        """Return lightweight molecule rows for selectors and tables."""
        query = """
            SELECT id, code, name, material_type_id, smiles, input_smiles, canonical_smiles, inchi, inchikey,
                   is_hidden, source, created_at, updated_at
            FROM molecules
        """
        parameters: list[object] = []
        filters = self._build_molecule_filters(
            keyword=keyword or search_text,
            include_hidden=include_hidden,
            hidden_only=hidden_only,
            parameter_filters=parameter_filters,
            material_type_id=material_type_id,
            parameters=parameters,
        )
        if filters:
            query += " WHERE " + " AND ".join(filters)
        sort_column = self._resolve_molecule_sort_column(sort_by)
        direction = self._resolve_sort_direction(descending)
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
        material_type_id: int | None = None,
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
            material_type_id=material_type_id,
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
            material_type_id=material_type_id,
        )
        return {"page": page, "page_size": page_size, "total": total, "items": items}

    def count_molecules(
        self,
        *,
        keyword: str | None = None,
        include_hidden: bool = False,
        hidden_only: bool = False,
        parameter_filters: dict[str, object] | None = None,
        material_type_id: int | None = None,
    ) -> int:
        """Count molecules matching the same filters as paginated listings."""
        query = "SELECT COUNT(*) AS count FROM molecules"
        parameters: list[object] = []
        filters = self._build_molecule_filters(
            keyword=keyword,
            include_hidden=include_hidden,
            hidden_only=hidden_only,
            parameter_filters=parameter_filters,
            material_type_id=material_type_id,
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
                SELECT id, code, name, material_type_id, smiles, input_smiles, canonical_smiles, inchi, inchikey,
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
            material_type_id=int(molecule_row["material_type_id"]) if molecule_row["material_type_id"] is not None else None,
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
            "material_type_id": detail.material_type_id,
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
            exists = connection.execute("SELECT id FROM molecules WHERE id = ?", (molecule_id,)).fetchone()
            if exists is None:
                return False
            self._delete_molecule_children(connection, molecule_id)
            cursor = connection.execute("DELETE FROM molecules WHERE id = ?", (molecule_id,))
            connection.commit()
            return cursor.rowcount > 0

    def _delete_molecule_children(self, connection: sqlite3.Connection, molecule_id: int) -> None:
        """Remove child rows explicitly for legacy databases without full cascade FKs."""
        connection.execute("UPDATE predictions SET molecule_id = NULL WHERE molecule_id = ?", (molecule_id,))
        connection.execute("UPDATE molecules SET material_type_id = NULL WHERE id = ?", (molecule_id,))
        connection.execute("DELETE FROM molecule_parameters WHERE molecule_id = ?", (molecule_id,))
        connection.execute("DELETE FROM molecule_descriptors WHERE molecule_id = ?", (molecule_id,))
        connection.execute("DELETE FROM molecular_features WHERE molecule_id = ?", (molecule_id,))
        connection.execute("DELETE FROM property_data WHERE molecule_id = ?", (molecule_id,))
        connection.execute("DELETE FROM lubricant_properties WHERE molecule_id = ?", (molecule_id,))
        connection.execute("DELETE FROM formula_components WHERE molecule_id = ?", (molecule_id,))
        connection.execute("DELETE FROM additive_compatibilities WHERE additive_id = ? OR base_oil_id = ?", (molecule_id, molecule_id))

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
                self._insert_descriptor_row(connection, molecule_id, descriptors, fingerprint_bits, now)
            else:
                self._update_descriptor_row(connection, molecule_id, descriptors, fingerprint_bits, now)
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
        if existing is None:
            self._insert_descriptor_row(connection, molecule_id, descriptors, fingerprint_bits, timestamp)
        else:
            self._update_descriptor_row(connection, molecule_id, descriptors, fingerprint_bits, timestamp)

    def _insert_descriptor_row(
        self,
        connection: sqlite3.Connection,
        molecule_id: int,
        descriptors: dict[str, object],
        fingerprint_bits: str,
        timestamp: str,
    ) -> None:
        payload = json.dumps(descriptors, ensure_ascii=False)
        if "descriptor_values" in self._list_columns(connection, "molecule_descriptors"):
            connection.execute(
                """
                INSERT INTO molecule_descriptors (
                    molecule_id, descriptor_values, descriptor_values_json,
                    fingerprint_bits, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (molecule_id, payload, payload, fingerprint_bits, timestamp, timestamp),
            )
            return

        connection.execute(
            """
            INSERT INTO molecule_descriptors (
                molecule_id, descriptor_values_json, fingerprint_bits, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (molecule_id, payload, fingerprint_bits, timestamp, timestamp),
        )

    def _update_descriptor_row(
        self,
        connection: sqlite3.Connection,
        molecule_id: int,
        descriptors: dict[str, object],
        fingerprint_bits: str,
        timestamp: str,
    ) -> None:
        payload = json.dumps(descriptors, ensure_ascii=False)
        if "descriptor_values" in self._list_columns(connection, "molecule_descriptors"):
            connection.execute(
                """
                UPDATE molecule_descriptors
                SET descriptor_values = ?, descriptor_values_json = ?, fingerprint_bits = ?, updated_at = ?
                WHERE molecule_id = ?
                """,
                (payload, payload, fingerprint_bits, timestamp, molecule_id),
            )
            return

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
            return pd.DataFrame(columns=["id", "name", "material_type_id", "smiles", "source", "created_at"])

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

        ordered_columns = ["id", "name", "material_type_id", "smiles", "source", "created_at"]
        remaining_columns = [column for column in dataset.columns if column not in ordered_columns]
        return dataset[ordered_columns + sorted(remaining_columns)]

    def save_formula(
        self,
        formula_name: str,
        composition: list[dict[str, object]],
        predicted_properties: dict[str, float],
        note: str = "",
        conditions: dict[str, float] | None = None,
        components: list[dict[str, object]] | None = None,
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
            formula_id = int(cursor.lastrowid)
            if components is not None:
                self._replace_formula_components(connection, formula_id, components)
            connection.commit()
            return formula_id

    def save_formulation(
        self,
        formula_name: str,
        note: str,
        composition: list[dict[str, object]],
        target_values: dict[str, float],
        conditions: dict[str, float] | None = None,
        components: list[dict[str, object]] | None = None,
    ) -> int:
        """Persist a formulation record used for formula learning and training."""
        return self.save_formula(
            formula_name=formula_name,
            note=note,
            composition=composition,
            conditions=conditions,
            predicted_properties=target_values,
            components=components,
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
        material_type_id: int | None,
        parameters: list[object],
    ) -> list[str]:
        """Build SQL snippets and append bound parameters for molecule filtering."""
        filters: list[str] = []
        if hidden_only:
            filters.append("is_hidden = 1")
        elif not include_hidden:
            filters.append("is_hidden = 0")

        if material_type_id is not None:
            filters.append("material_type_id = ?")
            parameters.append(int(material_type_id))

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

    @staticmethod
    def _resolve_sort_direction(descending: bool) -> str:
        """Resolve a public boolean flag to a safe SQL sort direction."""
        if not isinstance(descending, bool):
            raise ValueError("descending must be a boolean.")
        direction = "DESC" if descending else "ASC"
        if direction not in ALLOWED_SORT_DIRECTIONS:
            raise ValueError(f"Unsupported SQL sort direction: {direction}")
        return direction

    @staticmethod
    def _quote_identifier(identifier: str, allowed_values: set[str]) -> str:
        """Return a quoted SQLite identifier from an explicit allow-list."""
        if identifier not in allowed_values:
            raise ValueError(f"Unsupported SQL identifier: {identifier}")
        return f'"{identifier}"'

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

    @staticmethod
    def _clean_int(value: object) -> int | None:
        if value is None or value == "":
            return None
        return int(value)
