from __future__ import annotations

import sqlite3

import pytest

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.models import MoleculeImportRecord


def test_database_initializes(chemstudio_env):
    db_manager = DatabaseManager(chemstudio_env["database_path"])
    db_manager.initialize_database()
    assert db_manager.count_rows("molecules") == 0
    assert db_manager.count_rows("molecule_parameters") == 0
    assert db_manager.count_rows("molecule_descriptors") == 0
    assert db_manager.count_rows("model_records") == 0
    assert db_manager.count_rows("predictions") == 0


def test_database_rejects_unsafe_dynamic_sql_inputs(chemstudio_env):
    db_manager = DatabaseManager(chemstudio_env["database_path"])
    db_manager.initialize_database()

    with pytest.raises(ValueError, match="Unsupported SQL identifier"):
        db_manager.count_rows("molecules; DROP TABLE molecules")

    with pytest.raises(ValueError, match="descending must be a boolean"):
        db_manager.list_molecules(descending="DESC")


def test_database_connection_uses_wal_and_closes(chemstudio_env):
    db_manager = DatabaseManager(chemstudio_env["database_path"])
    db_manager.initialize_database()

    with db_manager.connect() as connection:
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        busy_timeout = int(connection.execute("PRAGMA busy_timeout").fetchone()[0])

    assert journal_mode == "wal"
    assert busy_timeout == 30000
    with pytest.raises(Exception, match="closed database"):
        connection.execute("SELECT 1")


def test_database_migrates_legacy_descriptor_values_column(tmp_path):
    database_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE molecule_descriptors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id INTEGER NOT NULL,
                descriptor_values JSON NOT NULL,
                fingerprint_bits TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            INSERT INTO molecule_descriptors (
                molecule_id, descriptor_values, fingerprint_bits, created_at, updated_at
            )
            VALUES (1, '{"MolWt": 18.02}', '', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
            """
        )

    db_manager = DatabaseManager(database_path)
    db_manager.initialize_database()

    with db_manager.connect() as connection:
        columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(molecule_descriptors)")}
        row = connection.execute(
            "SELECT descriptor_values_json FROM molecule_descriptors WHERE molecule_id = 1"
        ).fetchone()

    assert "descriptor_values_json" in columns
    assert row is not None
    assert row["descriptor_values_json"] == '{"MolWt": 18.02}'

    saved = db_manager.save_molecule({"name": "ethanol", "smiles": "CCO", "canonical_smiles": "CCO"})
    molecule_id = int(saved["id"])
    db_manager.save_descriptors(molecule_id, {"ABC": 1.25})
    with db_manager.connect() as connection:
        inserted = connection.execute(
            """
            SELECT descriptor_values, descriptor_values_json
            FROM molecule_descriptors
            WHERE molecule_id = ?
            """,
            (molecule_id,),
        ).fetchone()

    assert inserted is not None
    assert inserted["descriptor_values"] == '{"ABC": 1.25}'
    assert inserted["descriptor_values_json"] == '{"ABC": 1.25}'

    assert db_manager.delete_molecule(molecule_id) is True
    with db_manager.connect() as connection:
        descriptor_count = connection.execute(
            "SELECT COUNT(*) FROM molecule_descriptors WHERE molecule_id = ?",
            (molecule_id,),
        ).fetchone()[0]

    assert descriptor_count == 0


def test_database_migrates_legacy_unique_canonical_smiles_constraint(tmp_path):
    database_path = tmp_path / "legacy_unique.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE molecules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE,
                name TEXT NOT NULL,
                smiles TEXT,
                input_smiles TEXT NOT NULL DEFAULT '',
                canonical_smiles TEXT NOT NULL UNIQUE DEFAULT '',
                inchi TEXT NOT NULL DEFAULT '',
                inchikey TEXT NOT NULL DEFAULT '',
                molblock TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                is_hidden INTEGER NOT NULL DEFAULT 0,
                source TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE molecule_parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (molecule_id, key),
                FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE CASCADE
            );
            CREATE TABLE molecule_descriptors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id INTEGER NOT NULL UNIQUE,
                descriptor_values_json TEXT NOT NULL DEFAULT '{}',
                fingerprint_bits TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE CASCADE
            );
            CREATE TABLE molecular_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id INTEGER NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value REAL,
                FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE CASCADE
            );
            CREATE TABLE property_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id INTEGER NOT NULL,
                property_name TEXT NOT NULL,
                property_value REAL,
                FOREIGN KEY (molecule_id) REFERENCES molecules (id) ON DELETE CASCADE
            );
            """
        )

    db_manager = DatabaseManager(database_path)
    db_manager.initialize_database()

    first_id = db_manager.insert_molecule_record(
        MoleculeImportRecord(name="ethanol-a", smiles="CCO", canonical_smiles="CCO", code="S-001")
    )
    second_id = db_manager.insert_molecule_record(
        MoleculeImportRecord(name="ethanol-b", smiles="CCO", canonical_smiles="CCO", code="S-002")
    )

    assert first_id != second_id
    assert db_manager.count_rows("molecules") == 2


def test_database_migrates_lubricant_property_data_to_rich_table(tmp_path):
    database_path = tmp_path / "legacy_properties.sqlite"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE molecules (
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
            CREATE TABLE property_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                molecule_id INTEGER NOT NULL,
                property_name TEXT NOT NULL,
                property_value REAL
            );
            INSERT INTO molecules (id, code, name, smiles, source, created_at)
            VALUES (1, 'L-001', 'sample', 'CCO', '', '2026-01-01T00:00:00+00:00');
            INSERT INTO property_data (molecule_id, property_name, property_value)
            VALUES (1, 'wear_scar_width', 0.42);
            """
        )

    db_manager = DatabaseManager(database_path)
    db_manager.initialize_database()

    with db_manager.connect() as connection:
        row = connection.execute(
            """
            SELECT property_name, property_value, property_unit
            FROM lubricant_properties
            WHERE molecule_id = 1
            """
        ).fetchone()

    assert row is not None
    assert row["property_name"] == "wear_scar_width"
    assert row["property_value"] == 0.42
    assert row["property_unit"] == "mm"
