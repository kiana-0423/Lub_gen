"""Right-side inspector and chemistry action panel."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from draw.chemistry_services.base import ChemistryService
from draw.core.models import BondType, MoleculeDocument


class InspectorPanel(QWidget):
    """Displays editor state and chemistry entry points."""

    load_smiles_requested = Signal(str)
    generate_smiles_requested = Signal()
    generate_2d_requested = Signal()
    expand_explicit_hydrogens_requested = Signal()
    export_mol_requested = Signal()
    export_sdf_requested = Signal()
    export_pdb_requested = Signal()

    def __init__(self, chemistry_service: ChemistryService, document: MoleculeDocument | None = None) -> None:
        super().__init__()
        self._document = document or MoleculeDocument()
        self._backend_value = QLabel(chemistry_service.name)
        self._backend_status_value = QLabel(chemistry_service.describe())
        self._tool_value = QLabel("Select")
        self._element_value = QLabel("C")
        self._bond_value = QLabel(BondType.SINGLE.display_name)
        self._document_value = QLabel(self._format_document_summary(self._document))
        self._validation_value = QLabel(self._format_validation_summary(self._document))
        self._selection_value = QLabel("0 atoms, 0 bonds")
        self._smiles_input = QLineEdit()
        self._build_ui()

    def set_active_tool(self, tool_name: str) -> None:
        """Update the displayed active tool."""
        self._tool_value.setText(tool_name)

    def set_current_element(self, symbol: str) -> None:
        """Update the displayed current element."""
        self._element_value.setText(symbol)

    def set_current_bond_type(self, bond_type: BondType | str) -> None:
        """Update the displayed current bond type."""
        resolved = bond_type if isinstance(bond_type, BondType) else BondType(bond_type)
        self._bond_value.setText(resolved.display_name)

    def update_document(self, document: MoleculeDocument) -> None:
        """Update the displayed document and validation summaries."""
        self._document = document
        self._document_value.setText(self._format_document_summary(document))
        self._validation_value.setText(self._format_validation_summary(document))

    def update_selection(self, atom_count: int, bond_count: int) -> None:
        """Update the displayed selection summary."""
        self._selection_value.setText(f"{atom_count} atoms, {bond_count} bonds")

    def set_smiles_text(self, smiles: str) -> None:
        """Update the SMILES text field."""
        self._smiles_input.setText(smiles)

    def smiles_text(self) -> str:
        """Return the current SMILES input text."""
        return self._smiles_input.text()

    def _build_ui(self) -> None:
        self._backend_status_value.setWordWrap(True)
        self._document_value.setWordWrap(True)
        self._validation_value.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Backend:", self._backend_value)
        form.addRow("Backend status:", self._backend_status_value)
        form.addRow("Active tool:", self._tool_value)
        form.addRow("Current element:", self._element_value)
        form.addRow("Current bond:", self._bond_value)
        form.addRow("Document:", self._document_value)
        form.addRow("Validation:", self._validation_value)
        form.addRow("Selection:", self._selection_value)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addLayout(form)
        layout.addWidget(self._build_structure_io_group())
        layout.addStretch(1)

    @staticmethod
    def _format_document_summary(document: MoleculeDocument) -> str:
        return f"{document.atom_count} atoms, {document.bond_count} bonds"

    @staticmethod
    def _format_validation_summary(document: MoleculeDocument) -> str:
        invalid_atoms = sum(1 for atom in document.atoms.values() if not atom.is_valid)
        if invalid_atoms == 0:
            return "No atom-level validation warnings."
        return f"{invalid_atoms} atoms currently flagged by RDKit validation."

    def _build_structure_io_group(self) -> QGroupBox:
        group = QGroupBox("Structure I/O")
        layout = QVBoxLayout(group)

        self._smiles_input.setPlaceholderText("Enter SMILES and press Load")
        self._smiles_input.returnPressed.connect(self._emit_load_smiles)
        layout.addWidget(QLabel("SMILES"))
        layout.addWidget(self._smiles_input)

        smiles_row = QHBoxLayout()
        load_button = QPushButton("Load SMILES")
        load_button.clicked.connect(lambda checked=False: self._emit_load_smiles())
        generate_button = QPushButton("Generate SMILES")
        generate_button.clicked.connect(lambda checked=False: self.generate_smiles_requested.emit())
        smiles_row.addWidget(load_button)
        smiles_row.addWidget(generate_button)
        layout.addLayout(smiles_row)

        export_row = QHBoxLayout()
        export_mol_button = QPushButton("Export MOL")
        export_mol_button.clicked.connect(lambda checked=False: self.export_mol_requested.emit())
        export_sdf_button = QPushButton("Export SDF")
        export_sdf_button.clicked.connect(lambda checked=False: self.export_sdf_requested.emit())
        export_pdb_button = QPushButton("Export PDB")
        export_pdb_button.clicked.connect(lambda checked=False: self.export_pdb_requested.emit())
        export_row.addWidget(export_mol_button)
        export_row.addWidget(export_sdf_button)
        export_row.addWidget(export_pdb_button)
        layout.addLayout(export_row)

        layout_row = QHBoxLayout()
        generate_2d_button = QPushButton("Generate 2D")
        generate_2d_button.clicked.connect(lambda checked=False: self.generate_2d_requested.emit())
        expand_hydrogens_button = QPushButton("Expand H")
        expand_hydrogens_button.clicked.connect(
            lambda checked=False: self.expand_explicit_hydrogens_requested.emit()
        )
        layout_row.addWidget(generate_2d_button)
        layout_row.addWidget(expand_hydrogens_button)
        layout.addLayout(layout_row)

        hint = QLabel(
            "Implicit hydrogens and validation warnings are refreshed through the chemistry service after each edit."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return group

    def _emit_load_smiles(self) -> None:
        self.load_smiles_requested.emit(self._smiles_input.text())
