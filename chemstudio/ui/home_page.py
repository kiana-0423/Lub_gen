from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


class HomePage(QWidget):
    """Function-oriented landing page."""

    navigate_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(14)

        title_label = QLabel("ChemStudio")
        title_label.setStyleSheet("font-size: 26px; font-weight: 800;")

        subtitle_label = QLabel(
            "用于材料/化学数据管理、分子设计、配方设计与性能预测的模块化桌面应用。"
        )
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("font-size: 15px; color: #555555;")

        buttons_layout = QGridLayout()
        buttons_layout.setHorizontalSpacing(12)
        buttons_layout.setVerticalSpacing(12)
        buttons_layout.setColumnStretch(0, 1)
        buttons_layout.setColumnStretch(1, 1)
        buttons_layout.setRowStretch(0, 1)
        buttons_layout.setRowStretch(1, 1)

        entries = [
            ("数据导入与可视化", "data"),
            ("分子设计", "molecule"),
            ("分子编辑器", "editor"),
            ("配方设计", "formula"),
        ]
        for index, (label, page_key) in enumerate(entries):
            button = QPushButton(label)
            button.setMinimumHeight(82)
            button.setMinimumWidth(180)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            button.setStyleSheet(
                """
                QPushButton {
                    font-size: 18px;
                    font-weight: 700;
                    border-radius: 8px;
                    background: #f2f5f7;
                    border: 1px solid #d9e2ec;
                }
                QPushButton:hover {
                    background: #d9edf7;
                }
                """
            )
            button.clicked.connect(lambda _checked=False, key=page_key: self.navigate_requested.emit(key))
            buttons_layout.addWidget(button, index // 2, index % 2)

        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addSpacing(4)
        layout.addLayout(buttons_layout, stretch=1)
        scroll_area.setWidget(content)
        root_layout.addWidget(scroll_area)
