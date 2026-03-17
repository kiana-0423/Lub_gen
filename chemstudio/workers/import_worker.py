from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class ImportWorker(QThread):
    finished = Signal(dict)
    failed = Signal(str)

    def run(self) -> None:  # pragma: no cover
        self.finished.emit({"status": "not_implemented"})

