from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class HomePage(QWidget):
    """Function-oriented landing page."""

    navigate_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(24)

        title_label = QLabel("ChemStudio")
        title_label.setStyleSheet("font-size: 32px; font-weight: 800;")

        subtitle_label = QLabel(
            "用于材料/化学数据管理、分子设计、配方设计与性能预测的模块化桌面应用。"
        )
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("font-size: 15px; color: #555555;")

        buttons_layout = QGridLayout()
        buttons_layout.setHorizontalSpacing(18)
        buttons_layout.setVerticalSpacing(18)

        entries = [
            ("数据导入与可视化", "data"),
            ("分子设计", "molecule"),
            ("配方设计", "formula"),
        ]
        for column, (label, page_key) in enumerate(entries):
            button = QPushButton(label)
            button.setMinimumHeight(120)
            button.setStyleSheet(
                """
                QPushButton {
                    font-size: 18px;
                    font-weight: 700;
                    border-radius: 14px;
                    background: #f2f5f7;
                    border: 1px solid #d9e2ec;
                }
                QPushButton:hover {
                    background: #d9edf7;
                }
                """
            )
            button.clicked.connect(lambda _checked=False, key=page_key: self.navigate_requested.emit(key))
            buttons_layout.addWidget(button, 0, column)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(8)
        layout.addLayout(buttons_layout)
        layout.addStretch(1)
