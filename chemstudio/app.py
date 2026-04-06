from __future__ import annotations

import os
import sys

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.services.data_import_service import DataImportService
from chemstudio.utils.config import AppConfig, ensure_runtime_directories
from chemstudio.utils.logger import configure_logging


def configure_qt_runtime() -> None:
    """Apply safe default Qt settings for headless-friendly Linux execution."""
    if sys.platform.startswith("linux"):
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("QT_QUICK_BACKEND", "software")
        os.environ.setdefault("QSG_RHI_BACKEND", "software")

        existing_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        fallback_flags = ["--disable-gpu", "--disable-gpu-compositing"]
        merged = " ".join([existing_flags, *[flag for flag in fallback_flags if flag not in existing_flags]]).strip()
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = merged

        if hasattr(os, "geteuid") and os.geteuid() == 0:
            os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")


def bootstrap_database() -> DatabaseManager:
    """Create the SQLite database and load mock data if the database is empty."""
    db_manager = DatabaseManager(AppConfig.DATABASE_PATH)
    db_manager.initialize_database()
    DataImportService(db_manager).seed_mock_data_if_empty()
    return db_manager


def main() -> int:
    """Application entry point."""
    configure_qt_runtime()
    ensure_runtime_directories()
    configure_logging()
    db_manager = bootstrap_database()

    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("PySide6 is required to start the UI. Install dependencies from requirements.txt.") from exc

    from chemstudio.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName(AppConfig.APP_NAME)
    window = MainWindow(db_manager=db_manager)
    window.show()
    return app.exec()
