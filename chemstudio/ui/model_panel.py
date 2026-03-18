from __future__ import annotations

import json

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QPlainTextEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)


class ModelPanel(QWidget):
    refresh_requested = Signal()
    train_requested = Signal(dict)
    model_selected = Signal(int)
    predict_current_requested = Signal(dict)
    predict_batch_requested = Signal(dict)
    predict_manual_requested = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self._model_name = QLineEdit(self)
        self._model_type = QComboBox(self)
        self._feature_columns = QLineEdit(self)
        self._target_column = QLineEdit(self)
        self._include_hidden = QCheckBox("Include Hidden Training Rows", self)
        self._model_list = QListWidget(self)
        self._model_detail = QPlainTextEdit(self)
        self._manual_features = QPlainTextEdit(self)
        self._batch_ids = QLineEdit(self)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("Models", self)
        title.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(title)

        train_group = QGroupBox("Train", self)
        train_form = QFormLayout(train_group)
        self._model_name.setPlaceholderText("e.g. viscosity regressor")
        self._model_type.addItem("Random Forest Regressor", "random_forest_regressor")
        self._model_type.addItem("Linear Regression", "linear_regression")
        self._model_type.addItem("Random Forest Classifier", "random_forest_classifier")
        self._model_type.addItem("Logistic Regression", "logistic_regression")
        self._feature_columns.setPlaceholderText("parameter:molecular_weight, parameter:boiling_point")
        self._target_column.setPlaceholderText("parameter:target_score")
        train_form.addRow("Model Name", self._model_name)
        train_form.addRow("Model Type", self._model_type)
        train_form.addRow("Feature Columns", self._feature_columns)
        train_form.addRow("Target Column", self._target_column)
        train_form.addRow("", self._include_hidden)

        train_buttons = QHBoxLayout()
        train_button = QPushButton("Train", self)
        train_button.clicked.connect(self._emit_train_requested)
        refresh_button = QPushButton("Refresh Models", self)
        refresh_button.clicked.connect(self.refresh_requested.emit)
        train_buttons.addWidget(train_button)
        train_buttons.addWidget(refresh_button)
        train_form.addRow("", self._wrap_layout(train_buttons))
        layout.addWidget(train_group)

        model_group = QGroupBox("Saved Models", self)
        model_layout = QVBoxLayout(model_group)
        self._model_list.itemSelectionChanged.connect(self._on_model_selected)
        model_layout.addWidget(self._model_list, stretch=1)
        self._model_detail.setReadOnly(True)
        model_layout.addWidget(self._model_detail, stretch=1)
        layout.addWidget(model_group, stretch=1)

        predict_group = QGroupBox("Predict", self)
        predict_form = QFormLayout(predict_group)
        self._batch_ids.setPlaceholderText("Comma-separated molecule IDs, e.g. 1,2,3")
        self._manual_features.setPlaceholderText(
            '{"parameter:molecular_weight": 95.0, "parameter:boiling_point": 150.0}'
        )
        predict_form.addRow("Batch IDs", self._batch_ids)
        predict_form.addRow("Manual Features JSON", self._manual_features)

        predict_buttons = QHBoxLayout()
        predict_current = QPushButton("Predict Current Molecule", self)
        predict_current.clicked.connect(self._emit_predict_current)
        predict_batch = QPushButton("Predict Batch IDs", self)
        predict_batch.clicked.connect(self._emit_predict_batch)
        predict_manual = QPushButton("Predict Manual Features", self)
        predict_manual.clicked.connect(self._emit_predict_manual)
        predict_buttons.addWidget(predict_current)
        predict_buttons.addWidget(predict_batch)
        predict_buttons.addWidget(predict_manual)
        predict_form.addRow("", self._wrap_layout(predict_buttons))
        layout.addWidget(predict_group)

    def set_models(self, models: list[dict[str, object]], selected_model_id: int | None = None) -> None:
        self._model_list.clear()
        target_row = -1
        for index, model in enumerate(models):
            label = f'{model["id"]}: {model["name"]} [{model["model_type"]}]'
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, int(model["id"]))
            item.setData(Qt.ItemDataRole.UserRole + 1, model)
            self._model_list.addItem(item)
            if selected_model_id is not None and int(model["id"]) == selected_model_id:
                target_row = index

        if target_row >= 0:
            self._model_list.setCurrentRow(target_row)
        elif self._model_list.count() > 0:
            self._model_list.setCurrentRow(0)
        else:
            self._model_detail.clear()

    def selected_model_id(self) -> int | None:
        item = self._model_list.currentItem()
        if item is None:
            return None
        model_id = item.data(Qt.ItemDataRole.UserRole)
        return int(model_id) if model_id is not None else None

    def _on_model_selected(self) -> None:
        item = self._model_list.currentItem()
        if item is None:
            self._model_detail.clear()
            return
        model = item.data(Qt.ItemDataRole.UserRole + 1) or {}
        self._model_detail.setPlainText(json.dumps(model, indent=2, ensure_ascii=False, sort_keys=True))
        model_id = self.selected_model_id()
        if model_id is not None:
            self.model_selected.emit(model_id)

    def _emit_train_requested(self) -> None:
        feature_columns = [item.strip() for item in self._feature_columns.text().split(",") if item.strip()]
        payload = {
            "name": self._model_name.text().strip() or self._model_type.currentText(),
            "model_type": self._model_type.currentData(),
            "feature_columns": feature_columns,
            "target_column": self._target_column.text().strip(),
            "include_hidden": self._include_hidden.isChecked(),
        }
        self.train_requested.emit(payload)

    def _emit_predict_current(self) -> None:
        model_id = self.selected_model_id()
        if model_id is not None:
            self.predict_current_requested.emit({"model_id": model_id})

    def _emit_predict_batch(self) -> None:
        model_id = self.selected_model_id()
        if model_id is None:
            return

        molecule_ids: list[int] = []
        try:
            for chunk in self._batch_ids.text().split(","):
                text = chunk.strip()
                if text:
                    molecule_ids.append(int(text))
        except ValueError:
            QMessageBox.warning(self, "Invalid IDs", "Batch IDs must be comma-separated integers.")
            return

        self.predict_batch_requested.emit({"model_id": model_id, "molecule_ids": molecule_ids})

    def _emit_predict_manual(self) -> None:
        model_id = self.selected_model_id()
        if model_id is None:
            return

        raw = self._manual_features.toPlainText().strip() or "{}"
        try:
            feature_values = json.loads(raw)
        except json.JSONDecodeError as exc:
            QMessageBox.warning(self, "Invalid JSON", f"Manual features must be valid JSON: {exc.msg}")
            return

        if not isinstance(feature_values, dict):
            QMessageBox.warning(self, "Invalid JSON", "Manual features must be a JSON object.")
            return
        self.predict_manual_requested.emit({"model_id": model_id, "feature_values": feature_values})

    @staticmethod
    def _wrap_layout(layout) -> QWidget:
        container = QWidget()
        container.setLayout(layout)
        return container
