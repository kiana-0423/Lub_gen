from __future__ import annotations

import os
from pathlib import Path

from chemstudio.utils.file_utils import ensure_directory


class AppConfig:
    """Static runtime configuration."""

    APP_NAME = "ChemStudio"
    PACKAGE_DIR = Path(__file__).resolve().parents[1]
    PROJECT_ROOT = PACKAGE_DIR.parent
    RESOURCES_DIR = PACKAGE_DIR / "resources"
    DATABASE_PATH = Path(os.getenv("CHEMSTUDIO_DATABASE_PATH", str(RESOURCES_DIR / "chemstudio.sqlite")))
    SAMPLE_DATA_PATH = RESOURCES_DIR / "mock_materials.csv"
    SAVED_MODELS_DIR = Path(os.getenv("CHEMSTUDIO_MODEL_STORE_PATH", str(RESOURCES_DIR / "saved_models")))
    LOG_DIR = RESOURCES_DIR / "logs"
    WINDOW_WIDTH = 1480
    WINDOW_HEIGHT = 920

    @classmethod
    def database_path(cls) -> Path:
        """Return the effective SQLite database path."""
        return Path(os.getenv("CHEMSTUDIO_DATABASE_PATH", str(cls.DATABASE_PATH)))

    @classmethod
    def model_store_path(cls) -> Path:
        """Return the effective model storage directory."""
        return Path(os.getenv("CHEMSTUDIO_MODEL_STORE_PATH", str(cls.SAVED_MODELS_DIR)))


def ensure_runtime_directories() -> None:
    """Create runtime directories used by the application."""
    ensure_directory(AppConfig.RESOURCES_DIR)
    ensure_directory(AppConfig.model_store_path())
    ensure_directory(AppConfig.LOG_DIR)
