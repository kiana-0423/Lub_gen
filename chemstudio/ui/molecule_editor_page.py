from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.repositories import MoleculeRepository
from chemstudio.ui.molecule_editor_inspector import MoleculeEditorInspector
from chemstudio.ui.widgets import BasePage

from draw.chemistry_services import ChemistryServiceError, create_chemistry_service
from draw.commands import EditorCommandStack
from draw.core import COMMON_ELEMENT_SYMBOLS
from draw.core.models import BondType, MoleculeDocument
from draw.editor import ALL_TOOLS, EditorCanvas
from draw.ui.status_panel import StatusPanel
from draw.ui.tool_panel import ToolPanel


class MoleculeEditorPage(BasePage):
    """Embedded small-molecule editor with SMILES export and database import."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        molecule_repository: MoleculeRepository | None = None,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.molecule_repository = molecule_repository or MoleculeRepository(db_manager)
        self.chemistry_service = create_chemistry_service()
        self.command_stack = EditorCommandStack()
        self.document = MoleculeDocument()
        self.canvas = EditorCanvas(command_stack=self.command_stack, chemistry_service=self.chemistry_service)
        self.tool_panel = ToolPanel()
        self.inspector = MoleculeEditorInspector(
            chemistry_service=self.chemistry_service,
            document=self.document,
        )
        self.status_panel = StatusPanel()
        self._build_ui()
        self._connect_signals()
        self._sync_ui_state()

    def _build_ui(self) -> None:
        root_layout = self.create_page_shell(
            "分子编辑器",
            "绘制小分子结构，导入或生成 SMILES，并将结果保存到 ChemStudio 数据库。",
        )
        root_layout.addLayout(self._build_control_row())

        splitter = QSplitter()
        splitter.addWidget(self.tool_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.inspector)
        splitter.setSizes([220, 820, 320])

        root_layout.addWidget(splitter, stretch=1)
        root_layout.addWidget(QLabel("状态日志"))
        root_layout.addWidget(self.status_panel, stretch=0)

    def _build_control_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(8)

        for tool_name in ALL_TOOLS:
            button = QPushButton(tool_name)
            button.clicked.connect(lambda _checked=False, value=tool_name: self._set_active_tool(value))
            layout.addWidget(button)

        layout.addSpacing(12)
        for symbol in ("C", "N", "O", "S", "P", "Cl", "Br", "F"):
            if symbol not in COMMON_ELEMENT_SYMBOLS:
                continue
            button = QPushButton(symbol)
            button.clicked.connect(lambda _checked=False, value=symbol: self._set_current_element(value))
            layout.addWidget(button)

        layout.addSpacing(12)
        for bond_type in BondType:
            button = QPushButton(bond_type.display_name)
            button.clicked.connect(lambda _checked=False, value=bond_type: self._set_current_bond_type(value))
            layout.addWidget(button)

        layout.addSpacing(12)
        actions: list[tuple[str, Callable[[], None]]] = [
            ("撤销", self.command_stack.undo),
            ("重做", self.command_stack.redo),
            ("清除画布", self.canvas.clear_canvas),
            ("放大", self.canvas.zoom_in),
            ("缩小", self.canvas.zoom_out),
            ("生成 2D 坐标", self._generate_2d_coordinates),
            ("展开显式氢", self._expand_explicit_hydrogens),
            ("导入到数据库", self._import_to_database),
        ]
        for label, callback in actions:
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, handler=callback: handler())
            layout.addWidget(button)
        layout.addStretch(1)
        return layout

    def _connect_signals(self) -> None:
        self.tool_panel.tool_selected.connect(self._set_active_tool)
        self.tool_panel.element_selected.connect(self._set_current_element)
        self.tool_panel.bond_type_selected.connect(self._set_current_bond_type)
        self.canvas.status_message.connect(self._handle_status_message)
        self.canvas.document_changed.connect(self._on_document_changed)
        self.canvas.selection_summary_changed.connect(self.inspector.update_selection)
        self.inspector.load_smiles_requested.connect(self._load_smiles)
        self.inspector.generate_smiles_requested.connect(self._generate_smiles)
        self.inspector.generate_2d_requested.connect(self._generate_2d_coordinates)
        self.inspector.expand_explicit_hydrogens_requested.connect(self._expand_explicit_hydrogens)
        self.inspector.export_mol_requested.connect(self._export_mol)
        self.inspector.export_sdf_requested.connect(self._export_sdf)
        self.inspector.export_pdb_requested.connect(self._export_pdb)
        self.inspector.import_to_database_requested.connect(self._import_to_database)
        self.inspector.export_smiles_to_clipboard_requested.connect(
            lambda: self._handle_status_message("SMILES 已复制到剪贴板。")
        )

    def _sync_ui_state(self) -> None:
        self._set_active_tool(self.canvas.active_tool)
        self._set_current_element(self.canvas.current_element_symbol)
        self._set_current_bond_type(self.canvas.current_bond_type)
        self._on_document_changed(self.canvas.document_snapshot())
        self._handle_status_message(self.chemistry_service.describe())

    def _set_active_tool(self, tool_name: str) -> None:
        self.tool_panel.set_active_tool(tool_name)
        self.canvas.set_active_tool(tool_name)
        self.inspector.set_active_tool(tool_name)

    def _set_current_element(self, symbol: str) -> None:
        self.tool_panel.set_current_element(symbol)
        self.canvas.set_current_element(symbol)
        self.inspector.set_current_element(symbol)

    def _set_current_bond_type(self, bond_type: BondType | str) -> None:
        resolved = bond_type if isinstance(bond_type, BondType) else BondType(bond_type)
        self.tool_panel.set_current_bond_type(resolved)
        self.canvas.set_current_bond_type(resolved)
        self.inspector.set_current_bond_type(resolved)

    def _on_document_changed(self, document: MoleculeDocument) -> None:
        self.document = document
        self.inspector.update_document(document)
        self.inspector.set_imported_status(None)

    def _load_smiles(self, smiles: str) -> None:
        try:
            document = self.chemistry_service.import_smiles(smiles)
            canonical_smiles = self.chemistry_service.export_smiles(document)
        except ChemistryServiceError as exc:
            self._show_chemistry_error("SMILES 导入失败", str(exc))
            return

        self.command_stack.clear()
        self.canvas.load_document(document, fit_view=True)
        self.inspector.set_smiles_text(canonical_smiles)
        self._handle_status_message(f"已从 SMILES 载入结构: {canonical_smiles}")

    def _generate_smiles(self) -> None:
        try:
            smiles = self.chemistry_service.export_smiles(self.document)
        except ChemistryServiceError as exc:
            self._show_chemistry_error("SMILES 导出失败", str(exc))
            return
        self.inspector.set_smiles_text(smiles)
        self._handle_status_message(f"SMILES: {smiles}")

    def _generate_2d_coordinates(self) -> None:
        self._apply_document_transform(
            empty_message="空画布无法生成 2D 坐标。",
            error_title="2D 坐标生成失败",
            command_text="Generate 2D Coordinates",
            transform=self.chemistry_service.generate_2d_coordinates,
            success_message="已重新生成 2D 坐标。",
        )

    def _expand_explicit_hydrogens(self) -> None:
        self._apply_document_transform(
            empty_message="空画布无法展开显式氢。",
            error_title="显式氢展开失败",
            command_text="Expand Explicit Hydrogens",
            transform=self.chemistry_service.expand_explicit_hydrogens,
            success_message="已展开显式氢。",
        )

    def _import_to_database(self) -> None:
        if self.document.atom_count == 0:
            self._show_chemistry_error("导入失败", "当前画布为空。")
            return
        try:
            smiles = self.chemistry_service.export_smiles(self.document)
            molblock = self.chemistry_service.export_mol(self.document, molecule_name="molecule_editor")
        except ChemistryServiceError as exc:
            self._show_chemistry_error("导入失败", str(exc))
            return

        default_name = smiles or "editor_molecule"
        molecule_name, accepted = QInputDialog.getText(
            self,
            "导入到数据库",
            "分子名称:",
            text=default_name,
        )
        if not accepted:
            return
        molecule_name = molecule_name.strip() or default_name
        saved = self.molecule_repository.save_molecule(
            {
                "name": molecule_name,
                "smiles": smiles,
                "canonical_smiles": smiles,
                "molblock": molblock,
                "source": "molecule_editor",
            }
        )
        molecule_id = int(saved["id"])
        self.inspector.set_smiles_text(smiles)
        self.inspector.set_imported_status(molecule_id)
        self._handle_status_message(f"已导入数据库: {molecule_name} (ID: {molecule_id})")
        QMessageBox.information(self, "导入成功", f"已导入数据库，分子 ID = {molecule_id}")

    def _export_mol(self) -> None:
        self._save_chemistry_export(
            title="导出 MOL",
            default_suffix=".mol",
            file_filter="MDL Mol (*.mol);;All Files (*)",
            exporter=lambda name: self.chemistry_service.export_mol(self.document, molecule_name=name),
        )

    def _export_sdf(self) -> None:
        self._save_chemistry_export(
            title="导出 SDF",
            default_suffix=".sdf",
            file_filter="Structure Data File (*.sdf);;All Files (*)",
            exporter=lambda name: self.chemistry_service.export_sdf(self.document, molecule_name=name),
        )

    def _export_pdb(self) -> None:
        self._save_chemistry_export(
            title="导出 PDB",
            default_suffix=".pdb",
            file_filter="Protein Data Bank (*.pdb);;All Files (*)",
            exporter=lambda name: self.chemistry_service.export_pdb(self.document, molecule_name=name),
        )

    def _save_chemistry_export(
        self,
        *,
        title: str,
        default_suffix: str,
        file_filter: str,
        exporter: Callable[[str], str],
    ) -> None:
        if self.document.atom_count == 0:
            self._show_chemistry_error(f"{title}失败", "当前画布为空。")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, title, f"structure{default_suffix}", file_filter)
        if not file_path:
            return
        output_path = Path(file_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(default_suffix)
        try:
            output_path.write_text(exporter(output_path.stem), encoding="utf-8")
        except ChemistryServiceError as exc:
            self._show_chemistry_error(f"{title}失败", str(exc))
            return
        except OSError as exc:
            QMessageBox.warning(self, f"{title}失败", f"无法写入文件:\n{exc}")
            return
        self._handle_status_message(f"已导出结构到 {output_path.name}")

    def _apply_document_transform(
        self,
        *,
        empty_message: str,
        error_title: str,
        command_text: str,
        transform: Callable[[MoleculeDocument], MoleculeDocument],
        success_message: str,
    ) -> None:
        if self.document.atom_count == 0:
            self._show_chemistry_error(error_title, empty_message)
            return
        try:
            updated_document = transform(self.document)
        except ChemistryServiceError as exc:
            self._show_chemistry_error(error_title, str(exc))
            return
        self.canvas.apply_document_change(updated_document, command_text, fit_view=True)
        self._handle_status_message(success_message)

    def _handle_status_message(self, message: str) -> None:
        self.status_panel.append_message(message)

    def _show_chemistry_error(self, title: str, message: str) -> None:
        self._handle_status_message(message)
        QMessageBox.warning(self, title, message)
