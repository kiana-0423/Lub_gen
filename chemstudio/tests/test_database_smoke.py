from __future__ import annotations

import pytest

from chemstudio.database.db_manager import DatabaseManager


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
