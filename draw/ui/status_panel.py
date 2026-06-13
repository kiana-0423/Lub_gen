"""Bottom status and log panel."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import QPlainTextEdit


class StatusPanel(QPlainTextEdit):
    """Read-only log area for startup and editor status messages."""

    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(500)

    def append_message(self, message: str) -> None:
        """Append a timestamped message to the panel."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.appendPlainText(f"[{timestamp}] {message}")

