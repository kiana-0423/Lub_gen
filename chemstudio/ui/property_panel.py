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
        self._id_label = QLabel("-", self)
        self._code_label = QLabel("-", self)
        self._name_label = QLabel("-", self)
        self._smiles_label = QLabel("-", self)
        self._hidden_label = QLabel("-", self)
        self._parameters_text = self._build_readonly_text()
        self._descriptors_text = self._build_readonly_text()
        self._prediction_text = self._build_readonly_text()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("Details", self)
        title.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(title)

        info_group = QGroupBox("Molecule", self)
        form = QFormLayout(info_group)
        form.addRow("ID", self._id_label)
        form.addRow("Code", self._code_label)
        form.addRow("Name", self._name_label)
        form.addRow("SMILES", self._smiles_label)
        form.addRow("Hidden", self._hidden_label)
        layout.addWidget(info_group)

        parameters_group = QGroupBox("Parameters", self)
        parameters_layout = QVBoxLayout(parameters_group)
        parameters_layout.addWidget(self._parameters_text)
        layout.addWidget(parameters_group, stretch=1)

        descriptors_group = QGroupBox("Descriptors", self)
        descriptors_layout = QVBoxLayout(descriptors_group)
        descriptors_layout.addWidget(self._descriptors_text)
        layout.addWidget(descriptors_group, stretch=1)

        prediction_group = QGroupBox("Prediction", self)
        prediction_layout = QVBoxLayout(prediction_group)
        prediction_layout.addWidget(self._prediction_text)
        layout.addWidget(prediction_group, stretch=1)

    def clear(self) -> None:
        self._id_label.setText("-")
        self._code_label.setText("-")
        self._name_label.setText("-")
        self._smiles_label.setText("-")
        self._hidden_label.setText("-")
        self._parameters_text.clear()
        self._descriptors_text.clear()
        self._prediction_text.clear()

    def update_molecule(self, molecule: dict[str, object]) -> None:
        self._id_label.setText(str(molecule.get("id") or "-"))
        self._code_label.setText(str(molecule.get("code") or "-"))
        self._name_label.setText(str(molecule.get("name") or "-"))
        self._smiles_label.setText(str(molecule.get("canonical_smiles") or "-"))
        self._hidden_label.setText("Yes" if molecule.get("is_hidden") else "No")
        self._parameters_text.setPlainText(
            json.dumps(molecule.get("parameters") or {}, indent=2, ensure_ascii=False, sort_keys=True)
        )
        self.update_descriptors(molecule.get("descriptor_values") or {})

    def update_descriptors(self, descriptors: dict[str, object]) -> None:
        self._descriptors_text.setPlainText(json.dumps(descriptors, indent=2, ensure_ascii=False, sort_keys=True))

    def update_prediction(self, payload: dict[str, object] | None) -> None:
        if payload is None:
            self._prediction_text.clear()
            return
        self._prediction_text.setPlainText(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _build_readonly_text() -> QPlainTextEdit:
        widget = QPlainTextEdit()
        widget.setReadOnly(True)
        return widget
