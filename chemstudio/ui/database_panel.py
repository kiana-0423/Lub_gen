from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class DatabasePanel(QWidget):
    molecule_selected = Signal(int)
    filters_changed = Signal()
    refresh_requested = Signal()
    import_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._summary_label = QLabel("0 molecules", self)
        self._search_input = QLineEdit(self)
        self._include_hidden = QCheckBox("Include Hidden", self)
        self._hidden_only = QCheckBox("Hidden Only", self)
        self._list = QListWidget(self)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("Database", self)
        title.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(title)

        controls = QHBoxLayout()
        self._search_input.setPlaceholderText("Search name / code / SMILES / parameters")
        self._search_input.returnPressed.connect(self.filters_changed.emit)
        controls.addWidget(self._search_input, stretch=1)

        import_button = QPushButton("Import", self)
        import_button.clicked.connect(self.import_requested.emit)
        controls.addWidget(import_button)

        refresh_button = QPushButton("Refresh", self)
        refresh_button.clicked.connect(self.refresh_requested.emit)
        controls.addWidget(refresh_button)
        layout.addLayout(controls)

        toggles = QHBoxLayout()
        self._include_hidden.toggled.connect(self.filters_changed.emit)
        self._hidden_only.toggled.connect(self.filters_changed.emit)
        toggles.addWidget(self._include_hidden)
        toggles.addWidget(self._hidden_only)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        layout.addWidget(self._summary_label)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list, stretch=1)

    def current_filters(self) -> dict[str, object]:
        return {
            "keyword": self._search_input.text().strip() or None,
            "include_hidden": self._include_hidden.isChecked(),
            "hidden_only": self._hidden_only.isChecked(),
        }

    def set_molecules(self, listing: dict[str, object], selected_molecule_id: int | None = None) -> None:
        items = list(listing.get("items") or [])
        total = int(listing.get("total") or 0)
        self._summary_label.setText(f"{total} molecules")

        self._list.clear()
        target_row = -1
        for index, molecule in enumerate(items):
            parts = []
            if molecule.get("code"):
                parts.append(str(molecule["code"]))
            parts.append(str(molecule["name"]))
            if molecule.get("is_hidden"):
                parts.append("[hidden]")
            label = " ".join(parts)
            item = QListWidgetItem(label)
            item.setToolTip(str(molecule.get("canonical_smiles") or ""))
            item.setData(Qt.ItemDataRole.UserRole, int(molecule["id"]))
            self._list.addItem(item)
            if selected_molecule_id is not None and int(molecule["id"]) == selected_molecule_id:
                target_row = index

        if target_row >= 0:
            self._list.setCurrentRow(target_row)

    def selected_molecule_id(self) -> int | None:
        item = self._list.currentItem()
        if item is None:
            return None
        molecule_id = item.data(Qt.ItemDataRole.UserRole)
        return int(molecule_id) if molecule_id is not None else None

    def _on_selection_changed(self) -> None:
        molecule_id = self.selected_molecule_id()
        if molecule_id is not None:
            self.molecule_selected.emit(molecule_id)
