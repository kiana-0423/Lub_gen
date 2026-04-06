from __future__ import annotations

from pathlib import Path

from chemstudio.utils.file_utils import ensure_directory


class AppConfig:
    """Static runtime configuration."""

    APP_NAME = "ChemStudio"
    PACKAGE_DIR = Path(__file__).resolve().parents[1]
    PROJECT_ROOT = PACKAGE_DIR.parent
    RESOURCES_DIR = PACKAGE_DIR / "resources"
    DATABASE_PATH = RESOURCES_DIR / "chemstudio_mvp.sqlite"
    SAMPLE_DATA_PATH = RESOURCES_DIR / "mock_materials.csv"
    SAVED_MODELS_DIR = RESOURCES_DIR / "saved_models"
    LOG_DIR = RESOURCES_DIR / "logs"
    WINDOW_WIDTH = 1480
    WINDOW_HEIGHT = 920


def ensure_runtime_directories() -> None:
    """Create runtime directories used by the application."""
    ensure_directory(AppConfig.RESOURCES_DIR)
    ensure_directory(AppConfig.SAVED_MODELS_DIR)
    ensure_directory(AppConfig.LOG_DIR)
