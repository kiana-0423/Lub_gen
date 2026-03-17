from __future__ import annotations

import os
import sys

from chemstudio.data.db import initialize_database
from chemstudio.utils.logger import configure_logging


def configure_qt_runtime() -> None:
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


def main() -> int:
    configure_qt_runtime()

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from chemstudio.ui.main_window import MainWindow

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    configure_logging()
    initialize_database()

    app = QApplication(sys.argv)
    app.setApplicationName("chemstudio")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
