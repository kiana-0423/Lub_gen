from __future__ import annotations

from datetime import datetime

from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class LogPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._text = QPlainTextEdit(self)
        self._text.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)

    def append_message(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._text.appendPlainText(f"[{timestamp}] {message}")
