from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class BasePage(QWidget):
    """Base page with a consistent header and a return-home action."""

    home_requested = Signal()

    def create_page_shell(self, title: str, subtitle: str) -> QVBoxLayout:
        """Create a page root layout and attach a standard page header."""
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)
        root_layout.addLayout(self._create_header(title, subtitle))
        return root_layout

    def _create_header(self, title: str, subtitle: str) -> QHBoxLayout:
        back_button = QPushButton("返回首页")
        back_button.clicked.connect(lambda: self.home_requested.emit())

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 24px; font-weight: 700;")

        subtitle_label = QLabel(subtitle)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("color: #666666;")

        text_layout = QVBoxLayout()
        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)

        header_layout = QHBoxLayout()
        header_layout.addWidget(back_button)
        header_layout.addLayout(text_layout, stretch=1)
        return header_layout

    def refresh_page(self) -> None:
        """Hook for pages that need to refresh when they become active."""
        return
