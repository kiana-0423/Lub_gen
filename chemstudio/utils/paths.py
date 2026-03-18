from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    return project_root() / "chemstudio"


def database_path() -> Path:
    configured = os.getenv("CHEMSTUDIO_DATABASE_PATH")
    if configured:
        return Path(configured)
    return package_root() / "resources" / "chemstudio.sqlite"


def model_store_path() -> Path:
    configured = os.getenv("CHEMSTUDIO_MODEL_STORE_PATH")
    if configured:
        return Path(configured)
    return package_root() / "resources" / "models"
