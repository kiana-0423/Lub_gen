from __future__ import annotations

import os

import pytest


@pytest.fixture()
def chemstudio_env(tmp_path, monkeypatch):
    database_file = tmp_path / "chemstudio.sqlite"
    model_dir = tmp_path / "models"
    monkeypatch.setenv("CHEMSTUDIO_DATABASE_PATH", str(database_file))
    monkeypatch.setenv("CHEMSTUDIO_MODEL_STORE_PATH", str(model_dir))
    return {
        "database_path": database_file,
        "model_dir": model_dir,
    }
