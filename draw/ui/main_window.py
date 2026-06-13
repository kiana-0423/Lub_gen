"""Main application window."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QFileDialog, QDockWidget, QInputDialog, QMainWindow, QMessageBox, QToolBar

from draw.chemistry_services import ChemistryServiceError
from draw.chemistry_services.base import ChemistryService
from draw.commands.stack import EditorCommandStack
from draw.core import COMMON_ELEMENT_SYMBOLS
from draw.core.models import BondType, MoleculeDocument
from draw.editor import ALL_TOOLS, EditorCanvas

from .element_dialog import ElementDialog
from .inspector_panel import InspectorPanel
from .status_panel import StatusPanel
from .tool_panel import ToolPanel


class MainWindow(QMainWindow):
    """Top-level desktop shell for the native editor MVP."""

    def __init__(self, chemistry_service: ChemistryService, command_stack: EditorCommandStack) -> None:
        super().__init__()
        self._chemistry_service = chemistry_service
        self._command_stack = command_stack
        self._document = MoleculeDocument()
        self._canvas = EditorCanvas(command_stack=command_stack, chemistry_service=chemistry_service)
        self._tool_panel = ToolPanel()
        self._inspector_panel = InspectorPanel(chemistry_service=chemistry_service, document=self._document)
        self._status_panel = StatusPanel()

        self._tool_actions: dict[str, QAction] = {}
        self._element_actions: dict[str, QAction] = {}
        self._bond_actions: dict[BondType, QAction] = {}
        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)
        self._element_group = QActionGroup(self)
        self._element_group.setExclusive(True)
        self._bond_group = QActionGroup(self)
        self._bond_group.setExclusive(True)

        self._build_window()
        self._connect_signals()
        self._sync_ui_state()
        self._log_startup()

    def _build_window(self) -> None:
        self.setWindowTitle("Chem Drawing Smiles")
        self.resize(1380, 880)
        self.setCentralWidget(self._canvas)
        self._build_docks()
        self._build_menu_bar()
        self._build_toolbar()
        self.statusBar().showMessage("Application ready.")

    def _build_docks(self) -> None:
        left_dock = QDockWidget("Tools", self)
        left_dock.setWidget(self._tool_panel)
        left_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left_dock)

        right_dock = QDockWidget("Inspector", self)
        right_dock.setWidget(self._inspector_panel)
        right_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)

        bottom_dock = QDockWidget("Status", self)
        bottom_dock.setWidget(self._status_panel)
        bottom_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bottom_dock)

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        clear_action = QAction("Clear Canvas", self)
        clear_action.triggered.connect(self._canvas.clear_canvas)
        file_menu.addAction(clear_action)
        file_menu.addSeparator()

        export_mol_action = QAction("Export MOL...", self)
        export_mol_action.triggered.connect(self._export_mol)
        file_menu.addAction(export_mol_action)

        export_pdb_action = QAction("Export PDB...", self)
        export_pdb_action.triggered.connect(self._export_pdb)
        file_menu.addAction(export_pdb_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        undo_action = self._command_stack.create_undo_action(self, "Undo")
        redo_action = self._command_stack.create_redo_action(self, "Redo")
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()

        delete_action = QAction("Delete Selected", self)
        delete_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        delete_action.triggered.connect(self._canvas.delete_selected)
        edit_menu.addAction(delete_action)

        view_menu = self.menuBar().addMenu("&View")
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self._canvas.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self._canvas.zoom_out)
        view_menu.addAction(zoom_out_action)

        tool_menu = self.menuBar().addMenu("&Tools")
        for tool_name in ALL_TOOLS:
            action = QAction(tool_name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, name=tool_name: self._set_active_tool(name))
            tool_menu.addAction(action)
            self._tool_group.addAction(action)
            self._tool_actions[tool_name] = action

        element_menu = self.menuBar().addMenu("&Elements")
        for symbol in COMMON_ELEMENT_SYMBOLS:
            action = QAction(symbol, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=symbol: self._set_current_element(value))
            element_menu.addAction(action)
            self._element_group.addAction(action)
            self._element_actions[symbol] = action
        element_menu.addSeparator()
        more_elements_action = QAction("More Elements...", self)
        more_elements_action.triggered.connect(self._open_element_dialog)
        element_menu.addAction(more_elements_action)

        bond_menu = self.menuBar().addMenu("&Bonds")
        for bond_type in BondType:
            action = QAction(bond_type.display_name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked=False, value=bond_type: self._set_current_bond_type(value))
            bond_menu.addAction(action)
            self._bond_group.addAction(action)
            self._bond_actions[bond_type] = action

        chemistry_menu = self.menuBar().addMenu("&Chemistry")
        import_smiles_action = QAction("Import From SMILES...", self)
        import_smiles_action.triggered.connect(self._prompt_and_load_smiles)
        chemistry_menu.addAction(import_smiles_action)

        generate_smiles_action = QAction("Generate SMILES", self)
        generate_smiles_action.triggered.connect(self._generate_smiles)
        chemistry_menu.addAction(generate_smiles_action)

        chemistry_menu.addSeparator()

        generate_2d_action = QAction("Generate 2D Coordinates", self)
        generate_2d_action.triggered.connect(self._generate_2d_coordinates)
        chemistry_menu.addAction(generate_2d_action)

        expand_hydrogens_action = QAction("Expand Explicit Hydrogens", self)
        expand_hydrogens_action.triggered.connect(self._expand_explicit_hydrogens)
        chemistry_menu.addAction(expand_hydrogens_action)

        chemistry_menu.addSeparator()

        export_sdf_action = QAction("Export SDF...", self)
        export_sdf_action.triggered.connect(self._export_sdf)
        chemistry_menu.addAction(export_sdf_action)

        backend_action = QAction("Backend Status", self)
        backend_action.triggered.connect(self._show_backend_status)
        chemistry_menu.addAction(backend_action)

        help_menu = self.menuBar().addMenu("&Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        for tool_name in ALL_TOOLS:
            toolbar.addAction(self._tool_actions[tool_name])

        toolbar.addSeparator()

        for symbol in ("C", "N", "O", "S", "P", "Cl"):
            action = self._element_actions.get(symbol)
            if action is not None:
                toolbar.addAction(action)

        more_elements_action = QAction("More Elements", self)
        more_elements_action.triggered.connect(self._open_element_dialog)
        toolbar.addAction(more_elements_action)

        toolbar.addSeparator()

        for bond_type in BondType:
            toolbar.addAction(self._bond_actions[bond_type])

        toolbar.addSeparator()

        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self._canvas.delete_selected)
        toolbar.addAction(delete_action)

        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self._canvas.clear_canvas)
        toolbar.addAction(clear_action)

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self._canvas.zoom_in)
        toolbar.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self._canvas.zoom_out)
        toolbar.addAction(zoom_out_action)

        generate_2d_action = QAction("Generate 2D", self)
        generate_2d_action.triggered.connect(self._generate_2d_coordinates)
        toolbar.addAction(generate_2d_action)

        expand_hydrogens_action = QAction("Expand H", self)
        expand_hydrogens_action.triggered.connect(self._expand_explicit_hydrogens)
        toolbar.addAction(expand_hydrogens_action)

        backend_action = QAction("Backend", self)
        backend_action.triggered.connect(self._show_backend_status)
        toolbar.addAction(backend_action)

    def _connect_signals(self) -> None:
        self._tool_panel.tool_selected.connect(self._set_active_tool)
        self._tool_panel.element_selected.connect(self._set_current_element)
        self._tool_panel.bond_type_selected.connect(self._set_current_bond_type)
        self._canvas.status_message.connect(self._handle_status_message)
        self._canvas.document_changed.connect(self._handle_document_changed)
        self._canvas.selection_summary_changed.connect(self._inspector_panel.update_selection)
        self._inspector_panel.load_smiles_requested.connect(self._load_smiles)
        self._inspector_panel.generate_smiles_requested.connect(self._generate_smiles)
        self._inspector_panel.generate_2d_requested.connect(self._generate_2d_coordinates)
        self._inspector_panel.expand_explicit_hydrogens_requested.connect(self._expand_explicit_hydrogens)
        self._inspector_panel.export_mol_requested.connect(self._export_mol)
        self._inspector_panel.export_sdf_requested.connect(self._export_sdf)
        self._inspector_panel.export_pdb_requested.connect(self._export_pdb)

    def _sync_ui_state(self) -> None:
        self._tool_panel.set_active_tool(self._canvas.active_tool)
        self._inspector_panel.set_active_tool(self._canvas.active_tool)
        self._set_checked_action(self._tool_actions, self._tool_group, self._canvas.active_tool)

        self._tool_panel.set_current_element(self._canvas.current_element_symbol)
        self._inspector_panel.set_current_element(self._canvas.current_element_symbol)
        self._set_checked_action(self._element_actions, self._element_group, self._canvas.current_element_symbol)

        self._tool_panel.set_current_bond_type(self._canvas.current_bond_type)
        self._inspector_panel.set_current_bond_type(self._canvas.current_bond_type)
        self._set_checked_action(self._bond_actions, self._bond_group, self._canvas.current_bond_type)

        self._handle_document_changed(self._canvas.document_snapshot())

    def _set_active_tool(self, tool_name: str) -> None:
        self._tool_panel.set_active_tool(tool_name)
        self._canvas.set_active_tool(tool_name)
        self._inspector_panel.set_active_tool(tool_name)
        self._set_checked_action(self._tool_actions, self._tool_group, tool_name)

    def _set_current_element(self, symbol: str) -> None:
        self._tool_panel.set_current_element(symbol)
        self._canvas.set_current_element(symbol)
        self._inspector_panel.set_current_element(symbol)
        self._set_checked_action(self._element_actions, self._element_group, symbol)

    def _set_current_bond_type(self, bond_type: BondType | str) -> None:
        resolved = bond_type if isinstance(bond_type, BondType) else BondType(bond_type)
        self._tool_panel.set_current_bond_type(resolved)
        self._canvas.set_current_bond_type(resolved)
        self._inspector_panel.set_current_bond_type(resolved)
        self._set_checked_action(self._bond_actions, self._bond_group, resolved)

    def _handle_status_message(self, message: str) -> None:
        self.statusBar().showMessage(message, 3000)
        self._status_panel.append_message(message)

    def _handle_document_changed(self, document: MoleculeDocument) -> None:
        self._document = document
        self._inspector_panel.update_document(document)

    def _show_backend_status(self) -> None:
        QMessageBox.information(
            self,
            "Chemistry Backend",
            self._chemistry_service.describe(),
        )

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "About",
            (
                "Chem Drawing Smiles\n\n"
                "Native Python MVP for a chemical structure editor with element, bond type, "
                "and RDKit-backed validation support."
            ),
        )

    def _log_startup(self) -> None:
        self._status_panel.append_message("Editor window initialized.")
        self._status_panel.append_message(
            "Available tools: selection, atom placement, bond placement, delete, clear, zoom, undo/redo."
        )
        self._status_panel.append_message(
            "Element switching, bond type switching, implicit hydrogen refresh, and RDKit validation are enabled."
        )
        self._status_panel.append_message(self._chemistry_service.describe())

    def _prompt_and_load_smiles(self) -> None:
        smiles, accepted = QInputDialog.getText(
            self,
            "Import From SMILES",
            "SMILES:",
            text=self._inspector_panel.smiles_text(),
        )
        if not accepted:
            return
        self._inspector_panel.set_smiles_text(smiles)
        self._load_smiles(smiles)

    def _load_smiles(self, smiles: str) -> None:
        try:
            document = self._chemistry_service.import_smiles(smiles)
            canonical_smiles = self._chemistry_service.export_smiles(document)
        except ChemistryServiceError as exc:
            self._show_chemistry_error("SMILES Import Failed", str(exc))
            return

        self._command_stack.clear()
        self._canvas.load_document(document, fit_view=True)
        self._inspector_panel.set_smiles_text(canonical_smiles)
        self._handle_status_message(
            f"Loaded structure from SMILES with {document.atom_count} atoms and {document.bond_count} bonds."
        )

    def _generate_smiles(self) -> None:
        try:
            smiles = self._chemistry_service.export_smiles(self._document)
        except ChemistryServiceError as exc:
            self._show_chemistry_error("SMILES Export Failed", str(exc))
            return

        self._inspector_panel.set_smiles_text(smiles)
        self._handle_status_message("Generated canonical SMILES from the current drawing.")
        self._status_panel.append_message(f"SMILES: {smiles}")

    def _generate_2d_coordinates(self) -> None:
        self._apply_chemistry_document_change(
            empty_message="Cannot generate 2D coordinates for an empty drawing.",
            error_title="2D Coordinate Generation Failed",
            command_text="Generate 2D Coordinates",
            transform=self._chemistry_service.generate_2d_coordinates,
            success_message="Regenerated 2D coordinates for the current structure.",
            fit_view=True,
        )

    def _expand_explicit_hydrogens(self) -> None:
        self._apply_chemistry_document_change(
            empty_message="Cannot expand hydrogens for an empty drawing.",
            error_title="Explicit Hydrogen Expansion Failed",
            command_text="Expand Explicit Hydrogens",
            transform=self._chemistry_service.expand_explicit_hydrogens,
            success_message="Expanded explicit hydrogens with RDKit AddHs.",
            fit_view=True,
        )

    def _export_mol(self) -> None:
        self._save_chemistry_export(
            title="Export MOL",
            default_suffix=".mol",
            file_filter="MDL Mol (*.mol);;All Files (*)",
            exporter=lambda molecule_name: self._chemistry_service.export_mol(
                self._document,
                molecule_name=molecule_name,
            ),
        )

    def _export_sdf(self) -> None:
        self._save_chemistry_export(
            title="Export SDF",
            default_suffix=".sdf",
            file_filter="Structure Data File (*.sdf);;All Files (*)",
            exporter=lambda molecule_name: self._chemistry_service.export_sdf(
                self._document,
                molecule_name=molecule_name,
            ),
        )

    def _export_pdb(self) -> None:
        self._save_chemistry_export(
            title="Export PDB",
            default_suffix=".pdb",
            file_filter="Protein Data Bank (*.pdb);;All Files (*)",
            exporter=lambda molecule_name: self._chemistry_service.export_pdb(
                self._document,
                molecule_name=molecule_name,
            ),
        )

    def _save_chemistry_export(
        self,
        *,
        title: str,
        default_suffix: str,
        file_filter: str,
        exporter: Callable[[str], str],
    ) -> None:
        if self._document.atom_count == 0:
            self._show_chemistry_error(f"{title} Failed", "The current drawing is empty.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            f"structure{default_suffix}",
            file_filter,
        )
        if not file_path:
            return

        output_path = Path(file_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(default_suffix)

        try:
            exported_text = exporter(output_path.stem)
            output_path.write_text(exported_text, encoding="utf-8")
        except ChemistryServiceError as exc:
            self._show_chemistry_error(f"{title} Failed", str(exc))
            return
        except OSError as exc:
            QMessageBox.warning(self, f"{title} Failed", f"Could not write the file:\n{exc}")
            return

        self._handle_status_message(f"Exported structure to {output_path.name}.")

    def _apply_chemistry_document_change(
        self,
        *,
        empty_message: str,
        error_title: str,
        command_text: str,
        transform: Callable[[MoleculeDocument], MoleculeDocument],
        success_message: str,
        fit_view: bool,
    ) -> None:
        if self._document.atom_count == 0:
            self._show_chemistry_error(error_title, empty_message)
            return

        try:
            updated_document = transform(self._document)
        except ChemistryServiceError as exc:
            self._show_chemistry_error(error_title, str(exc))
            return

        self._canvas.apply_document_change(updated_document, command_text, fit_view=fit_view)
        self._handle_status_message(success_message)

    def _open_element_dialog(self) -> None:
        dialog = ElementDialog(self._canvas.current_element_symbol, self)
        if dialog.exec():
            self._set_current_element(dialog.selected_symbol)

    def _show_chemistry_error(self, title: str, message: str) -> None:
        self._handle_status_message(message)
        QMessageBox.warning(self, title, message)

    @staticmethod
    def _set_checked_action(
        actions: dict[str, QAction] | dict[BondType, QAction],
        group: QActionGroup,
        key: str | BondType,
    ) -> None:
        action = actions.get(key)
        if action is None:
            MainWindow._clear_action_group(group)
            return
        action.setChecked(True)

    @staticmethod
    def _clear_action_group(group: QActionGroup) -> None:
        group.setExclusive(False)
        for action in group.actions():
            action.setChecked(False)
        group.setExclusive(True)
