from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class PropertyPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.name_label = QLabel("-", self)
        self.smiles_label = QLabel("-", self)
        self.identifier_label = QLabel("-", self)
        self.descriptor_text = QPlainTextEdit(self)
        self.descriptor_text.setReadOnly(True)
        self.prediction_text = QPlainTextEdit(self)
        self.prediction_text.setReadOnly(True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        molecule_group = QGroupBox("Molecule Info", self)
        form = QFormLayout(molecule_group)
        form.addRow("Name", self.name_label)
        form.addRow("SMILES", self.smiles_label)
        form.addRow("Identifier", self.identifier_label)
        layout.addWidget(molecule_group)

        descriptor_group = QGroupBox("Descriptors", self)
        descriptor_layout = QVBoxLayout(descriptor_group)
        descriptor_layout.addWidget(self.descriptor_text)
        layout.addWidget(descriptor_group, stretch=1)

        prediction_group = QGroupBox("Predictions", self)
        prediction_layout = QVBoxLayout(prediction_group)
        prediction_layout.addWidget(self.prediction_text)
        layout.addWidget(prediction_group, stretch=1)

    def update_molecule_fields(self, payload: dict) -> None:
        self.name_label.setText(payload.get("name") or "Untitled")
        self.smiles_label.setText(payload.get("smiles") or "-")
        identifier = str(payload.get("id", "-"))
        self.identifier_label.setText(identifier)

    def update_saved_molecule(self, molecule) -> None:
        self.name_label.setText(molecule.display_name)
        self.smiles_label.setText(molecule.canonical_smiles or "-")
        self.identifier_label.setText(str(molecule.id))

    def update_descriptors(self, descriptors: dict) -> None:
        self.descriptor_text.setPlainText(json.dumps(descriptors, indent=2, sort_keys=True))

