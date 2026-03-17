from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from chemstudio.bridges.editor_bridge import EditorBridge
from chemstudio.services.descriptor_service import DescriptorService
from chemstudio.services.molecule_service import MoleculeService
from chemstudio.ui.log_panel import LogPanel
from chemstudio.ui.molecule_editor_widget import MoleculeEditorWidget
from chemstudio.ui.project_panel import ProjectPanel
from chemstudio.ui.property_panel import PropertyPanel
from chemstudio.workers.descriptor_worker import DescriptorWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("chemstudio")
        self.resize(1600, 960)

        self.log_panel = LogPanel()
        self.project_panel = ProjectPanel()
        self.property_panel = PropertyPanel()
        self.molecule_service = MoleculeService()
        self.descriptor_service = DescriptorService()

        self.editor_bridge = EditorBridge(
            molecule_service=self.molecule_service,
            descriptor_service=self.descriptor_service,
        )
        self.editor_widget = MoleculeEditorWidget(self.editor_bridge)

        self._descriptor_worker: DescriptorWorker | None = None
        self._build_ui()
        self._connect_signals()
        self._load_initial_state()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        top_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        top_splitter.addWidget(self.project_panel)
        top_splitter.addWidget(self.editor_widget)
        top_splitter.addWidget(self.property_panel)
        top_splitter.setSizes([240, 820, 420])

        content = QWidget(self)
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(top_splitter)

        root_layout.addWidget(content, stretch=8)
        root_layout.addWidget(self.log_panel, stretch=2)
        self.setCentralWidget(central)

        self._build_toolbar()

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction("Save Molecule", self._save_current_molecule)
        toolbar.addAction("Compute Descriptors", self._compute_descriptors)
        toolbar.addAction("Refresh List", self._refresh_molecule_list)

    def _connect_signals(self) -> None:
        self.project_panel.molecule_selected.connect(self._load_molecule_into_editor)
        self.project_panel.refresh_requested.connect(self._refresh_molecule_list)
        self.editor_bridge.structureChanged.connect(self._on_structure_changed)
        self.editor_bridge.moleculeSaved.connect(self._on_molecule_saved)
        self.editor_bridge.descriptorsReady.connect(self._on_descriptors_ready)
        self.editor_bridge.logMessage.connect(self.log_panel.append_message)

    def _load_initial_state(self) -> None:
        self._refresh_molecule_list()
        self.log_panel.append_message("chemstudio initialized.")

    def _refresh_molecule_list(self) -> None:
        molecules = self.molecule_service.list_molecules()
        self.project_panel.set_molecules(molecules)
        self.log_panel.append_message(f"Loaded {len(molecules)} molecules from database.")

    def _on_structure_changed(self, payload: dict) -> None:
        self.property_panel.update_molecule_fields(payload)

    def _save_current_molecule(self) -> None:
        try:
            payload = self.editor_widget.request_structure_payload()
            molecule = self.molecule_service.save_molecule(payload)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Molecule", str(exc))
            self.log_panel.append_message(f"Save failed: {exc}")
            return

        self.property_panel.update_saved_molecule(molecule)
        self.editor_bridge.update_payload(
            {
                "id": molecule.id,
                "name": molecule.display_name,
                "smiles": molecule.canonical_smiles,
                "molblock": molecule.molblock or "",
            }
        )
        self.editor_bridge.emit_molecule_saved(self.editor_bridge.get_current_payload())
        self._refresh_molecule_list()
        self.log_panel.append_message(f"Saved molecule #{molecule.id}: {molecule.display_name}.")

    def _compute_descriptors(self) -> None:
        payload = self.editor_widget.request_structure_payload()
        smiles = payload.get("smiles", "").strip()
        if not smiles:
            QMessageBox.information(self, "No Structure", "Draw or enter a SMILES string first.")
            return

        self.log_panel.append_message("Starting descriptor calculation in background worker.")
        molecule_id = payload.get("id")
        self._descriptor_worker = DescriptorWorker(self.descriptor_service, smiles, molecule_id=molecule_id)
        self._descriptor_worker.result_ready.connect(self._on_worker_finished)
        self._descriptor_worker.failed.connect(self._on_worker_failed)
        self._descriptor_worker.start()

    def _on_worker_finished(self, descriptors: dict) -> None:
        self.property_panel.update_descriptors(descriptors)
        self.log_panel.append_message("Descriptor calculation completed.")
        self.editor_bridge.emit_descriptors(descriptors)
        self._descriptor_worker = None

    def _on_worker_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Descriptor Error", message)
        self.log_panel.append_message(f"Descriptor calculation failed: {message}")
        self._descriptor_worker = None

    def _load_molecule_into_editor(self, molecule_id: int) -> None:
        molecule = self.molecule_service.get_molecule(molecule_id)
        if molecule is None:
            self.log_panel.append_message(f"Molecule #{molecule_id} not found.")
            return

        payload = {
            "id": molecule.id,
            "name": molecule.display_name,
            "smiles": molecule.canonical_smiles,
            "molblock": molecule.molblock or "",
        }
        self.editor_widget.load_structure(payload)
        self.property_panel.update_saved_molecule(molecule)
        if molecule.descriptor_record is not None:
            self.property_panel.update_descriptors(molecule.descriptor_record.descriptor_values)
        else:
            self.property_panel.update_descriptors({})
        self.log_panel.append_message(f"Loaded molecule #{molecule.id} into editor.")

    def _on_molecule_saved(self, payload: dict) -> None:
        self.property_panel.update_molecule_fields(payload)

    def _on_descriptors_ready(self, descriptors: dict) -> None:
        self.property_panel.update_descriptors(descriptors)
