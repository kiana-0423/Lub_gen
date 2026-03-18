from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class MoleculeEditorWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._current_id: int | None = None
        self._code_input = QLineEdit(self)
        self._name_input = QLineEdit(self)
        self._smiles_input = QLineEdit(self)
        self._hidden_checkbox = QCheckBox("Hidden", self)
        self._molblock_input = QPlainTextEdit(self)
        self._notes_input = QPlainTextEdit(self)
        self._parameters_input = QPlainTextEdit(self)
        self._build_ui()
        self.clear_form()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("Molecule Editor", self)
        title.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(title)

        basic_group = QGroupBox("Basic Fields", self)
        basic_form = QFormLayout(basic_group)
        self._code_input.setPlaceholderText("Optional code, e.g. M-001")
        self._name_input.setPlaceholderText("Optional name")
        self._smiles_input.setPlaceholderText("SMILES, e.g. CCO")
        basic_form.addRow("Code", self._code_input)
        basic_form.addRow("Name", self._name_input)
        basic_form.addRow("SMILES", self._smiles_input)
        basic_form.addRow("", self._hidden_checkbox)
        layout.addWidget(basic_group)

        self._parameters_input.setPlaceholderText('{"boiling_point": 78.3, "target_score": 46.0}')
        parameters_group = QGroupBox("Parameters JSON", self)
        parameters_layout = QVBoxLayout(parameters_group)
        parameters_layout.addWidget(self._parameters_input)
        layout.addWidget(parameters_group, stretch=1)

        self._notes_input.setPlaceholderText("Free-text notes")
        notes_group = QGroupBox("Notes", self)
        notes_layout = QVBoxLayout(notes_group)
        notes_layout.addWidget(self._notes_input)
        layout.addWidget(notes_group, stretch=1)

        self._molblock_input.setPlaceholderText("Optional MolBlock")
        molblock_group = QGroupBox("MolBlock", self)
        molblock_layout = QVBoxLayout(molblock_group)
        molblock_layout.addWidget(self._molblock_input)
        layout.addWidget(molblock_group, stretch=1)

    def current_molecule_id(self) -> int | None:
        return self._current_id

    def clear_form(self) -> None:
        self.load_molecule(
            {
                "id": None,
                "code": "",
                "name": "",
                "canonical_smiles": "",
                "is_hidden": False,
                "notes": "",
                "molblock": "",
                "parameters": {},
            }
        )

    def load_molecule(self, molecule: dict[str, object]) -> None:
        self._current_id = int(molecule["id"]) if molecule.get("id") is not None else None
        self._code_input.setText(str(molecule.get("code") or ""))
        self._name_input.setText(str(molecule.get("name") or ""))
        self._smiles_input.setText(str(molecule.get("canonical_smiles") or molecule.get("input_smiles") or ""))
        self._hidden_checkbox.setChecked(bool(molecule.get("is_hidden", False)))
        self._notes_input.setPlainText(str(molecule.get("notes") or ""))
        self._molblock_input.setPlainText(str(molecule.get("molblock") or ""))
        parameters = molecule.get("parameters") or {}
        self._parameters_input.setPlainText(json.dumps(parameters, indent=2, ensure_ascii=False, sort_keys=True))

    def request_payload(self) -> dict[str, object]:
        return {
            "id": self._current_id,
            "code": self._empty_to_none(self._code_input.text()),
            "name": self._name_input.text().strip(),
            "smiles": self._smiles_input.text().strip(),
            "is_hidden": self._hidden_checkbox.isChecked(),
            "notes": self._notes_input.toPlainText(),
            "molblock": self._molblock_input.toPlainText(),
            "parameters": self._parse_parameters(),
        }

    def _parse_parameters(self) -> dict[str, object]:
        raw = self._parameters_input.toPlainText().strip()
        if not raw:
            return {}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Parameters must be valid JSON: {exc.msg}") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Parameters must be a JSON object.")
        return parsed

    @staticmethod
    def _empty_to_none(value: str) -> str | None:
        text = value.strip()
        return text or None
