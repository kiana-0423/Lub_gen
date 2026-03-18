from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from chemstudio.services.descriptor_service import DescriptorService
from chemstudio.services.import_service import ImportFileService
from chemstudio.services.model_service import ModelService
from chemstudio.services.molecule_service import MoleculeService
from chemstudio.ui.database_panel import DatabasePanel
from chemstudio.ui.log_panel import LogPanel
from chemstudio.ui.model_panel import ModelPanel
from chemstudio.ui.molecule_editor_widget import MoleculeEditorWidget
from chemstudio.ui.property_panel import PropertyPanel
from chemstudio.workers.descriptor_worker import DescriptorWorker
from chemstudio.workers.import_worker import ImportWorker
from chemstudio.workers.predict_worker import PredictWorker
from chemstudio.workers.train_worker import TrainWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChemStudio")
        self.resize(1700, 980)

        self.molecule_service = MoleculeService()
        self.descriptor_service = DescriptorService()
        self.import_service = ImportFileService()
        self.model_service = ModelService()

        self.database_panel = DatabasePanel()
        self.editor_widget = MoleculeEditorWidget()
        self.property_panel = PropertyPanel()
        self.model_panel = ModelPanel()
        self.log_panel = LogPanel()

        self._import_worker: ImportWorker | None = None
        self._descriptor_worker: DescriptorWorker | None = None
        self._train_worker: TrainWorker | None = None
        self._predict_worker: PredictWorker | None = None
        self._selected_molecule_id: int | None = None
        self._selected_model_id: int | None = None

        self._build_ui()
        self._connect_signals()
        self._load_initial_state()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        top_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        top_splitter.addWidget(self.database_panel)
        top_splitter.addWidget(self.editor_widget)
        top_splitter.addWidget(self.property_panel)
        top_splitter.setSizes([320, 760, 520])

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        bottom_splitter.addWidget(self.model_panel)
        bottom_splitter.addWidget(self.log_panel)
        bottom_splitter.setSizes([900, 700])

        root_splitter = QSplitter(Qt.Orientation.Vertical, self)
        root_splitter.addWidget(top_splitter)
        root_splitter.addWidget(bottom_splitter)
        root_splitter.setSizes([620, 320])

        root_layout.addWidget(root_splitter)
        self.setCentralWidget(central)

        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction("New", self._new_molecule)
        toolbar.addAction("Import", self._import_data)
        toolbar.addAction("Save", self._save_current_molecule)
        toolbar.addAction("Delete", self._delete_current_molecule)
        toolbar.addAction("Toggle Hidden", self._toggle_hidden)
        toolbar.addAction("Compute Descriptors", self._compute_descriptors)
        toolbar.addAction("Refresh", self._refresh_everything)

    def _connect_signals(self) -> None:
        self.database_panel.molecule_selected.connect(self._load_molecule)
        self.database_panel.filters_changed.connect(self._refresh_molecule_list)
        self.database_panel.import_requested.connect(self._import_data)
        self.database_panel.refresh_requested.connect(self._refresh_molecule_list)

        self.model_panel.refresh_requested.connect(self._refresh_models)
        self.model_panel.model_selected.connect(self._on_model_selected)
        self.model_panel.train_requested.connect(self._start_training)
        self.model_panel.predict_current_requested.connect(self._predict_current_molecule)
        self.model_panel.predict_batch_requested.connect(self._predict_batch)
        self.model_panel.predict_manual_requested.connect(self._predict_manual)

    def _load_initial_state(self) -> None:
        self._refresh_everything()
        self.log_panel.append_message("ChemStudio UI initialized.")

    def _refresh_everything(self) -> None:
        self._refresh_molecule_list()
        self._refresh_models()

    def _refresh_molecule_list(self) -> None:
        listing = self.molecule_service.list_molecules(page=1, page_size=500, **self.database_panel.current_filters())
        self.database_panel.set_molecules(listing, selected_molecule_id=self._selected_molecule_id)
        self.log_panel.append_message(f"Loaded {listing['total']} molecules.")

    def _refresh_models(self) -> None:
        models = self.model_service.list_models()
        self.model_panel.set_models(models, selected_model_id=self._selected_model_id)
        self.log_panel.append_message(f"Loaded {len(models)} models.")

    def _load_molecule(self, molecule_id: int) -> None:
        detail = self.molecule_service.get_molecule_detail(molecule_id)
        if detail is None:
            self.log_panel.append_message(f"Molecule {molecule_id} not found.")
            return

        self._selected_molecule_id = molecule_id
        self.editor_widget.load_molecule(detail)
        self.property_panel.update_molecule(detail)
        self.property_panel.update_prediction(None)
        self.statusBar().showMessage(f"Loaded molecule #{molecule_id}", 3000)

    def _new_molecule(self) -> None:
        self._selected_molecule_id = None
        self.editor_widget.clear_form()
        self.property_panel.clear()
        self.statusBar().showMessage("Ready to create a new molecule.", 3000)

    def _save_current_molecule(self) -> None:
        try:
            payload = self.editor_widget.request_payload()
            molecule_id = payload.pop("id")
            saved = self.molecule_service.save_molecule(payload, molecule_id=molecule_id)
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))
            self.log_panel.append_message(f"Save failed: {exc}")
            return

        self._selected_molecule_id = int(saved["id"])
        self._refresh_molecule_list()
        self._load_molecule(self._selected_molecule_id)
        self.log_panel.append_message(f"Saved molecule #{saved['id']} {saved['name']}.")

    def _import_data(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Molecule Data",
            "",
            "Supported Files (*.json *.csv *.xlsx *.xls);;JSON Files (*.json);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
        )
        if not file_path:
            return

        try:
            records = self.import_service.load_records(file_path)
        except Exception as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            self.log_panel.append_message(f"Import preparation failed: {exc}")
            return

        if not records:
            QMessageBox.information(self, "No Records", "No importable rows were found in the selected file.")
            self.log_panel.append_message("Import skipped because the selected file contained no usable rows.")
            return

        self._import_worker = ImportWorker(self.molecule_service, records)
        self._import_worker.result_ready.connect(self._on_import_ready)
        self._import_worker.failed.connect(self._on_worker_failed)
        self._import_worker.start()
        self.log_panel.append_message(f"Import started from {file_path} with {len(records)} rows.")

    def _on_import_ready(self, result: dict) -> None:
        items = list(result.get("items") or [])
        if items:
            self._selected_molecule_id = int(items[-1]["id"])
        self._refresh_molecule_list()
        if self._selected_molecule_id is not None:
            self._load_molecule(self._selected_molecule_id)
        self.log_panel.append_message(f"Import completed: {result.get('count', 0)} molecules saved.")
        self._import_worker = None

    def _delete_current_molecule(self) -> None:
        if self._selected_molecule_id is None:
            QMessageBox.information(self, "No Selection", "Select a molecule first.")
            return

        confirmed = QMessageBox.question(
            self,
            "Delete Molecule",
            f"Delete molecule #{self._selected_molecule_id}?",
        )
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        deleted = self.molecule_service.delete_molecule(self._selected_molecule_id)
        if deleted:
            self.log_panel.append_message(f"Deleted molecule #{self._selected_molecule_id}.")
            self._selected_molecule_id = None
            self.editor_widget.clear_form()
            self.property_panel.clear()
            self._refresh_molecule_list()

    def _toggle_hidden(self) -> None:
        if self._selected_molecule_id is None:
            QMessageBox.information(self, "No Selection", "Select a molecule first.")
            return

        detail = self.molecule_service.get_molecule_detail(self._selected_molecule_id)
        if detail is None:
            return
        updated = self.molecule_service.set_hidden_state(self._selected_molecule_id, not bool(detail["is_hidden"]))
        if updated is None:
            return
        self.log_panel.append_message(
            f"Molecule #{self._selected_molecule_id} hidden state set to {updated['is_hidden']}."
        )
        self._refresh_molecule_list()
        self._load_molecule(self._selected_molecule_id)

    def _compute_descriptors(self) -> None:
        try:
            payload = self.editor_widget.request_payload()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Input", str(exc))
            return

        smiles = str(payload.get("smiles") or "").strip()
        if not smiles:
            QMessageBox.information(self, "No Structure", "Enter a SMILES string first.")
            return

        self._descriptor_worker = DescriptorWorker(
            self.descriptor_service,
            smiles,
            molecule_id=self._selected_molecule_id,
        )
        self._descriptor_worker.result_ready.connect(self._on_descriptors_ready)
        self._descriptor_worker.failed.connect(self._on_worker_failed)
        self._descriptor_worker.start()
        self.log_panel.append_message("Descriptor calculation started.")

    def _on_descriptors_ready(self, descriptors: dict) -> None:
        self.property_panel.update_descriptors(descriptors)
        if self._selected_molecule_id is not None:
            detail = self.molecule_service.get_molecule_detail(self._selected_molecule_id)
            if detail is not None:
                self.property_panel.update_molecule(detail)
        self.log_panel.append_message("Descriptor calculation completed.")
        self._descriptor_worker = None

    def _start_training(self, training_kwargs: dict) -> None:
        if not training_kwargs.get("feature_columns"):
            QMessageBox.warning(self, "Missing Features", "Provide at least one feature column.")
            return
        if not training_kwargs.get("target_column"):
            QMessageBox.warning(self, "Missing Target", "Provide a target column.")
            return

        self._train_worker = TrainWorker(self.model_service, training_kwargs)
        self._train_worker.result_ready.connect(self._on_training_ready)
        self._train_worker.failed.connect(self._on_worker_failed)
        self._train_worker.start()
        self.log_panel.append_message(f"Training started for {training_kwargs['name']}.")

    def _on_training_ready(self, result: dict) -> None:
        model = result["model"]
        self._selected_model_id = int(model["id"])
        self._refresh_models()
        self.property_panel.update_prediction({"training_result": result})
        self.log_panel.append_message(
            f"Training finished for model #{model['id']} with {result['rows']} rows."
        )
        self._train_worker = None

    def _on_model_selected(self, model_id: int) -> None:
        self._selected_model_id = model_id

    def _predict_current_molecule(self, payload: dict) -> None:
        if self._selected_molecule_id is None:
            QMessageBox.information(self, "No Selection", "Select a molecule first.")
            return
        self._predict_worker = PredictWorker(
            self.model_service,
            {"model_id": int(payload["model_id"]), "molecule_ids": [self._selected_molecule_id]},
        )
        self._predict_worker.result_ready.connect(self._on_prediction_ready)
        self._predict_worker.failed.connect(self._on_worker_failed)
        self._predict_worker.start()
        self.log_panel.append_message(f"Prediction started for molecule #{self._selected_molecule_id}.")

    def _predict_batch(self, payload: dict) -> None:
        molecule_ids = list(payload.get("molecule_ids") or [])
        if not molecule_ids:
            QMessageBox.information(self, "No IDs", "Provide one or more molecule IDs.")
            return

        self._predict_worker = PredictWorker(
            self.model_service,
            {"model_id": int(payload["model_id"]), "molecule_ids": molecule_ids},
        )
        self._predict_worker.result_ready.connect(self._on_prediction_ready)
        self._predict_worker.failed.connect(self._on_worker_failed)
        self._predict_worker.start()
        self.log_panel.append_message(f"Batch prediction started for {len(molecule_ids)} molecules.")

    def _predict_manual(self, payload: dict) -> None:
        try:
            prediction_kwargs = {
                "model_id": int(payload["model_id"]),
                "feature_values": dict(payload["feature_values"]),
            }
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Features", str(exc))
            return

        self._predict_worker = PredictWorker(self.model_service, prediction_kwargs, single=True)
        self._predict_worker.result_ready.connect(self._on_prediction_ready)
        self._predict_worker.failed.connect(self._on_worker_failed)
        self._predict_worker.start()
        self.log_panel.append_message("Manual prediction started.")

    def _on_prediction_ready(self, result: dict) -> None:
        if "prediction" in result:
            self.property_panel.update_prediction(result)
            self.log_panel.append_message("Single prediction completed.")
        else:
            self.property_panel.update_prediction(result)
            self.log_panel.append_message(
                f"Batch prediction completed with {len(result.get('predictions') or [])} rows."
            )
        self._predict_worker = None

    def _on_worker_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Worker Error", message)
        self.log_panel.append_message(f"Worker failed: {message}")
        self._import_worker = None
        self._descriptor_worker = None
        self._train_worker = None
        self._predict_worker = None
