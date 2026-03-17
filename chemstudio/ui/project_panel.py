from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ProjectPanel(QWidget):
    molecule_selected = Signal(int)
    refresh_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._list = QListWidget(self)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel("Projects / Molecules", self)
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(header)

        button_row = QHBoxLayout()
        refresh = QPushButton("Refresh", self)
        refresh.clicked.connect(self.refresh_requested.emit)
        button_row.addWidget(refresh)
        layout.addLayout(button_row)

        self._list.itemClicked.connect(self._emit_selected)
        layout.addWidget(self._list)

    def set_molecules(self, molecules: list) -> None:
        self._list.clear()
        for molecule in molecules:
            label = f"{molecule.id}: {molecule.display_name}"
            item = QListWidgetItem(label)
            item.setData(32, molecule.id)
            self._list.addItem(item)

    def _emit_selected(self, item: QListWidgetItem) -> None:
        molecule_id = item.data(32)
        if molecule_id is not None:
            self.molecule_selected.emit(int(molecule_id))

