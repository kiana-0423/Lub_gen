from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from draw.ui.inspector_panel import InspectorPanel


class MoleculeEditorInspector(InspectorPanel):
    """ChemStudio-specific inspector actions for the embedded molecule editor."""

    import_to_database_requested = Signal()
    export_smiles_to_clipboard_requested = Signal()

    def __init__(self, *args, **kwargs) -> None:
        self._imported_status_label = QLabel("未导入数据库")
        self._material_type_hint_label = QLabel("")
        super().__init__(*args, **kwargs)

    def set_imported_status(self, molecule_id: int | None) -> None:
        if molecule_id is None:
            self._imported_status_label.setText("未导入数据库")
            return
        self._imported_status_label.setText(f"已导入数据库 (ID: {molecule_id})")

    def set_material_type_hint(self, type_name: str) -> None:
        self._material_type_hint_label.setText(f"当前分子类型: {type_name}" if type_name else "")

    def _build_structure_io_group(self):
        group = super()._build_structure_io_group()
        layout = group.layout()
        if layout is None:
            return group

        import_button = QPushButton("导入到数据库")
        import_button.clicked.connect(lambda _checked=False: self.import_to_database_requested.emit())

        copy_button = QPushButton("复制 SMILES")
        copy_button.clicked.connect(self._copy_smiles_to_clipboard)

        self._imported_status_label.setWordWrap(True)
        self._material_type_hint_label.setWordWrap(True)
        layout.addWidget(import_button)
        layout.addWidget(copy_button)
        layout.addWidget(self._imported_status_label)
        layout.addWidget(self._material_type_hint_label)
        return group

    def _copy_smiles_to_clipboard(self) -> None:
        QApplication.clipboard().setText(self.smiles_text())
        self.export_smiles_to_clipboard_requested.emit()
