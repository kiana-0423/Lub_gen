from __future__ import annotations

from chemstudio.database.db_manager import DatabaseManager


def test_database_initializes(chemstudio_env):
    db_manager = DatabaseManager(chemstudio_env["database_path"])
    db_manager.initialize_database()
    assert db_manager.count_rows("molecules") == 0
    assert db_manager.count_rows("molecule_parameters") == 0
    assert db_manager.count_rows("molecule_descriptors") == 0
    assert db_manager.count_rows("model_records") == 0
    assert db_manager.count_rows("predictions") == 0
