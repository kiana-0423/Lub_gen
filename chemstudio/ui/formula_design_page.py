from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.repositories import MoleculeRepository
from chemstudio.services.formula_service import FormulaService
from chemstudio.services.model_service import ModelService
from chemstudio.ui.widgets import BasePage, SHAPForceWidget, SHAPSummaryWidget


logger = logging.getLogger(__name__)


class FormulaTrainingWorker(QObject):
    """Run formulation model training outside the GUI thread."""

    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, formula_service: FormulaService, parameters: dict[str, Any]) -> None:
        super().__init__()
        self.formula_service = formula_service
        self.parameters = parameters

    @Slot()
    def run(self) -> None:
        try:
            artifact = self.formula_service.train_formulation_model(**self.parameters)
        except (RuntimeError, ValueError) as exc:
            logger.exception("Background formulation model training failed")
            self.failed.emit(str(exc))
            return
        self.finished.emit(artifact)


class FormulaExplanationWorker(QObject):
    """Run formulation SHAP model explanation outside the GUI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, model_service: ModelService, artifact: dict[str, Any]) -> None:
        super().__init__()
        self.model_service = model_service
        self.artifact = artifact

    @Slot()
    def run(self) -> None:
        try:
            explanation = self.model_service.explain_model(self.artifact)
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.exception("Background formulation model explanation failed")
            self.failed.emit(str(exc))
            return
        self.finished.emit(explanation)


class FormulaDesignPage(BasePage):
    """Formula-design module with an entry page, formulation learning, and ML prediction."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        formula_service: FormulaService,
        model_service: ModelService,
        molecule_repository: MoleculeRepository | None = None,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.molecule_repository = molecule_repository or MoleculeRepository(db_manager)
        self.formula_service = formula_service
        self.model_service = model_service
        self.current_artifact: dict[str, Any] | None = None
        self.current_explanation: Any | None = None
        self.molecule_catalog: list[dict[str, Any]] = []
        self._training_thread: QThread | None = None
        self._training_worker: FormulaTrainingWorker | None = None
        self._explanation_thread: QThread | None = None
        self._explanation_worker: FormulaExplanationWorker | None = None
        self._build_ui()
        self.refresh_page()

    def _build_ui(self) -> None:
        root_layout = self.create_page_shell(
            "配方设计",
            "先选择配方功能，再进行配方录入学习或基于已保存配方训练并预测性能。",
        )

        self.formula_stack = QStackedWidget()
        self.entry_page = self._build_entry_page()
        self.learning_page = self._build_learning_page()
        self.ml_page = self._build_ml_page()

        self.formula_stack.addWidget(self.entry_page)
        self.formula_stack.addWidget(self.learning_page)
        self.formula_stack.addWidget(self.ml_page)

        root_layout.addWidget(self.formula_stack, stretch=1)

    def _build_entry_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addStretch(1)

        title = QLabel("选择配方设计功能")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 700;")

        subtitle = QLabel("配方录入用于沉淀训练数据，机器学习模型页用于训练与预测。")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666666;")

        button_row = QHBoxLayout()
        button_row.setSpacing(16)

        learn_button = QPushButton("配方录入与学习")
        learn_button.setMinimumHeight(88)
        learn_button.clicked.connect(self._show_learning_page)

        model_button = QPushButton("机器学习配方生成模型")
        model_button.setMinimumHeight(88)
        model_button.clicked.connect(self._show_ml_page)

        button_row.addStretch(1)
        button_row.addWidget(learn_button, stretch=1)
        button_row.addWidget(model_button, stretch=1)
        button_row.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(button_row)
        layout.addStretch(2)
        return page

    def _build_learning_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        back_button = QPushButton("返回功能选择")
        back_button.clicked.connect(self._show_entry_page)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        content_layout.addWidget(self._build_learning_editor_panel(), stretch=3)
        content_layout.addWidget(self._build_saved_formulation_panel(), stretch=2)

        layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(content_layout, stretch=1)
        return page

    def _build_learning_editor_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        info_box = QGroupBox("配方基础信息")
        info_layout = QFormLayout(info_box)
        self.learning_formula_name_input = QLineEdit()
        self.learning_formula_name_input.setPlaceholderText("输入配方名称")
        self.learning_formula_note_input = QTextEdit()
        self.learning_formula_note_input.setPlaceholderText("输入配方说明或备注")
        self.learning_formula_note_input.setMaximumHeight(90)
        self.learning_test_conditions_input = QTextEdit()
        self.learning_test_conditions_input.setMaximumHeight(96)
        self.learning_test_conditions_input.setPlaceholderText(
            '输入 JSON 或 key=value 测试条件，例如:\n{"temperature": 25, "pressure": 1.0}\n或\ntemperature=25\npressure=1.0'
        )
        info_layout.addRow("配方名称", self.learning_formula_name_input)
        info_layout.addRow("配方说明", self.learning_formula_note_input)
        info_layout.addRow("测试条件", self.learning_test_conditions_input)

        targets_box = QGroupBox("目标性能")
        targets_layout = QGridLayout(targets_box)
        self.learning_target_inputs: dict[str, QLineEdit] = {}
        for row, field_name in enumerate(self.formula_service.DEFAULT_TARGET_FIELDS):
            field_input = QLineEdit()
            field_input.setPlaceholderText("可留空")
            self.learning_target_inputs[field_name] = field_input
            targets_layout.addWidget(QLabel(field_name), row, 0)
            targets_layout.addWidget(field_input, row, 1)

        components_box = QGroupBox("配方组分编辑")
        components_layout = QVBoxLayout(components_box)
        component_actions = QHBoxLayout()
        add_button = QPushButton("添加组分行")
        add_button.clicked.connect(lambda: self._insert_component_row(self.learning_component_table))
        remove_button = QPushButton("删除选中行")
        remove_button.clicked.connect(
            lambda: self._remove_selected_component_row(self.learning_component_table, self.learning_ratio_summary_label)
        )
        normalize_button = QPushButton("自动归一化")
        normalize_button.clicked.connect(
            lambda: self._normalize_component_table(self.learning_component_table, self.learning_ratio_summary_label)
        )
        clear_button = QPushButton("清空当前配方")
        clear_button.clicked.connect(self._clear_learning_editor)
        component_actions.addWidget(add_button)
        component_actions.addWidget(remove_button)
        component_actions.addWidget(normalize_button)
        component_actions.addWidget(clear_button)
        component_actions.addStretch(1)

        self.learning_component_table = self._create_component_table()
        self.learning_ratio_summary_label = QLabel("当前组分数: 0 | 比例总和: 0.0000")

        save_button = QPushButton("保存配方")
        save_button.clicked.connect(self._save_learning_formulation)

        components_layout.addLayout(component_actions)
        components_layout.addWidget(self.learning_component_table, stretch=1)
        components_layout.addWidget(self.learning_ratio_summary_label)
        components_layout.addWidget(save_button, alignment=Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(info_box)
        layout.addWidget(targets_box)
        layout.addWidget(components_box, stretch=1)
        return panel

    def _build_saved_formulation_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        saved_box = QGroupBox("已保存配方")
        saved_layout = QVBoxLayout(saved_box)

        self.saved_formulation_table = QTableWidget(0, 4)
        self.saved_formulation_table.setHorizontalHeaderLabels(["ID", "名称", "目标字段", "创建时间"])
        self.saved_formulation_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.saved_formulation_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.saved_formulation_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.saved_formulation_table.itemSelectionChanged.connect(self._display_selected_formulation)

        action_row = QHBoxLayout()
        refresh_button = QPushButton("刷新列表")
        refresh_button.clicked.connect(self._refresh_saved_formulations)
        load_button = QPushButton("载入到编辑区")
        load_button.clicked.connect(self._load_selected_formulation_to_editor)
        delete_button = QPushButton("删除选中配方")
        delete_button.clicked.connect(self._delete_selected_formulation)
        action_row.addWidget(refresh_button)
        action_row.addWidget(load_button)
        action_row.addWidget(delete_button)
        action_row.addStretch(1)

        self.saved_formulation_detail = QTextEdit()
        self.saved_formulation_detail.setReadOnly(True)

        saved_layout.addWidget(self.saved_formulation_table, stretch=1)
        saved_layout.addLayout(action_row)
        saved_layout.addWidget(self.saved_formulation_detail, stretch=1)

        layout.addWidget(saved_box, stretch=1)
        return panel

    def _build_ml_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        back_button = QPushButton("返回功能选择")
        back_button.clicked.connect(self._show_entry_page)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        content_layout.addWidget(self._build_training_panel(), stretch=2)
        content_layout.addWidget(self._build_prediction_panel(), stretch=3)

        layout.addWidget(back_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(content_layout, stretch=1)
        return page

    def _build_training_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        setup_box = QGroupBox("模型训练")
        setup_layout = QFormLayout(setup_box)
        self.training_target_combo = QComboBox()
        self.training_model_combo = QComboBox()
        self.training_train_button = QPushButton("训练模型")
        self.training_train_button.clicked.connect(self._train_formulation_model)
        setup_layout.addRow("目标字段", self.training_target_combo)
        setup_layout.addRow("模型", self.training_model_combo)
        setup_layout.addRow("", self.training_train_button)

        metrics_box = QGroupBox("训练结果")
        metrics_layout = QFormLayout(metrics_box)
        self.training_summary_label = QLabel("尚未训练")
        self.training_r2_label = QLabel("-")
        self.training_mae_label = QLabel("-")
        self.training_rmse_label = QLabel("-")
        self.training_model_info_label = QLabel("-")
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 0)
        self.training_progress.setVisible(False)
        self.formula_explain_button = QPushButton("模型解释")
        self.formula_explain_button.clicked.connect(self._explain_model)
        self.formula_explain_button.setEnabled(False)
        metrics_layout.addRow("数据概况", self.training_summary_label)
        metrics_layout.addRow("模型信息", self.training_model_info_label)
        metrics_layout.addRow("R²", self.training_r2_label)
        metrics_layout.addRow("MAE", self.training_mae_label)
        metrics_layout.addRow("RMSE", self.training_rmse_label)
        metrics_layout.addRow("训练状态", self.training_progress)
        metrics_layout.addRow("", self.formula_explain_button)

        self.formula_shap_summary_widget = SHAPSummaryWidget()
        self.formula_shap_summary_widget.setVisible(False)

        layout.addWidget(setup_box)
        layout.addWidget(metrics_box)
        layout.addWidget(self.formula_shap_summary_widget, stretch=1)
        layout.addStretch(1)
        self._update_explain_button_state()
        return panel

    def _build_prediction_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        predictor_box = QGroupBox("新配方预测")
        predictor_layout = QVBoxLayout(predictor_box)

        actions = QHBoxLayout()
        add_button = QPushButton("添加组分行")
        add_button.clicked.connect(lambda: self._insert_component_row(self.prediction_component_table))
        remove_button = QPushButton("删除选中行")
        remove_button.clicked.connect(
            lambda: self._remove_selected_component_row(
                self.prediction_component_table,
                self.prediction_ratio_summary_label,
            )
        )
        normalize_button = QPushButton("自动归一化")
        normalize_button.clicked.connect(
            lambda: self._normalize_component_table(
                self.prediction_component_table,
                self.prediction_ratio_summary_label,
            )
        )
        predict_button = QPushButton("预测")
        predict_button.clicked.connect(self._predict_new_formulation)
        actions.addWidget(add_button)
        actions.addWidget(remove_button)
        actions.addWidget(normalize_button)
        actions.addWidget(predict_button)
        actions.addStretch(1)

        self.prediction_component_table = self._create_component_table()
        self.prediction_ratio_summary_label = QLabel("当前组分数: 0 | 比例总和: 0.0000")
        self.prediction_test_conditions_input = QTextEdit()
        self.prediction_test_conditions_input.setMaximumHeight(96)
        self.prediction_test_conditions_input.setPlaceholderText(
            '输入 JSON 或 key=value 测试条件，例如:\n{"temperature": 25, "pressure": 1.0}\n或\ntemperature=25\npressure=1.0'
        )
        self.prediction_result_text = QTextEdit()
        self.prediction_result_text.setReadOnly(True)
        self.formula_shap_force_widget = SHAPForceWidget()
        self.formula_shap_force_widget.setVisible(False)

        predictor_layout.addLayout(actions)
        predictor_layout.addWidget(self.prediction_component_table, stretch=1)
        predictor_layout.addWidget(self.prediction_ratio_summary_label)
        predictor_layout.addWidget(QLabel("测试条件"))
        predictor_layout.addWidget(self.prediction_test_conditions_input)
        predictor_layout.addWidget(self.prediction_result_text, stretch=1)
        predictor_layout.addWidget(self.formula_shap_force_widget, stretch=1)

        layout.addWidget(predictor_box, stretch=1)
        return panel

    def refresh_page(self) -> None:
        self._refresh_molecule_catalog()
        self._refresh_saved_formulations()
        self._refresh_training_targets()
        self._refresh_model_catalog()
        self._refresh_component_table_catalog(self.learning_component_table, self.learning_ratio_summary_label)
        self._refresh_component_table_catalog(self.prediction_component_table, self.prediction_ratio_summary_label)

    def _show_entry_page(self) -> None:
        self.formula_stack.setCurrentWidget(self.entry_page)

    def _show_learning_page(self) -> None:
        self.formula_stack.setCurrentWidget(self.learning_page)
        self.refresh_page()

    def _show_ml_page(self) -> None:
        self.formula_stack.setCurrentWidget(self.ml_page)
        self.refresh_page()

    def _refresh_molecule_catalog(self) -> None:
        self.molecule_catalog = self.molecule_repository.list_catalog()

    def _refresh_training_targets(self) -> None:
        current_target = self.training_target_combo.currentText()
        self.training_target_combo.clear()
        for target_name in self.formula_service.get_available_target_fields():
            self.training_target_combo.addItem(target_name)
        if current_target:
            index = self.training_target_combo.findText(current_target)
            if index >= 0:
                self.training_target_combo.setCurrentIndex(index)

    def _refresh_model_catalog(self) -> None:
        current_key = self.training_model_combo.currentData()
        self.training_model_combo.clear()
        for item in self.formula_service.get_model_catalog():
            label = str(item["label"])
            if not item["available"]:
                label = f"{label} (未安装)"
            self.training_model_combo.addItem(label, userData=item["key"])
        if current_key is not None:
            index = self.training_model_combo.findData(current_key)
            if index >= 0:
                self.training_model_combo.setCurrentIndex(index)

    def _create_component_table(self) -> QTableWidget:
        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["分子", "SMILES", "比例", "备注"])
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _make_molecule_combo(self, table: QTableWidget) -> QComboBox:
        combo = QComboBox()
        combo.addItem("请选择分子", userData=None)
        for molecule in self.molecule_catalog:
            combo.addItem(f"{molecule['id']} | {molecule['name']}", userData=molecule)
        combo.currentIndexChanged.connect(lambda: self._handle_molecule_changed(table, combo))
        return combo

    def _insert_component_row(
        self,
        table: QTableWidget,
        *,
        molecule_id: int | None = None,
        ratio: float = 0.0,
        note: str = "",
    ) -> None:
        row = table.rowCount()
        table.insertRow(row)

        combo = self._make_molecule_combo(table)
        table.setCellWidget(row, 0, combo)

        smiles_item = QTableWidgetItem("")
        smiles_item.setFlags(smiles_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(row, 1, smiles_item)

        ratio_spin = QDoubleSpinBox()
        ratio_spin.setDecimals(4)
        ratio_spin.setRange(0.0, 100000.0)
        ratio_spin.setValue(float(ratio))
        ratio_spin.valueChanged.connect(lambda _value: self._update_ratio_summary_for_table(table))
        table.setCellWidget(row, 2, ratio_spin)

        note_item = QTableWidgetItem(note)
        table.setItem(row, 3, note_item)

        if molecule_id is not None:
            selected_index = next(
                (
                    index
                    for index in range(combo.count())
                    if combo.itemData(index) is not None and int(combo.itemData(index)["id"]) == int(molecule_id)
                ),
                0,
            )
            combo.setCurrentIndex(selected_index)

        self._update_ratio_summary_for_table(table)

    def _handle_molecule_changed(self, table: QTableWidget, combo: QComboBox) -> None:
        row = self._find_widget_row(table, combo)
        if row < 0:
            return
        molecule = combo.currentData()
        smiles_text = str(molecule.get("smiles") or "") if isinstance(molecule, dict) else ""
        table.item(row, 1).setText(smiles_text)

    def _find_widget_row(self, table: QTableWidget, widget: QWidget) -> int:
        for row in range(table.rowCount()):
            if table.cellWidget(row, 0) is widget or table.cellWidget(row, 2) is widget:
                return row
        return -1

    def _remove_selected_component_row(self, table: QTableWidget, summary_label: QLabel) -> None:
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)
        self._update_ratio_summary(table, summary_label)

    def _snapshot_component_table(self, table: QTableWidget) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for row in range(table.rowCount()):
            combo = table.cellWidget(row, 0)
            ratio_spin = table.cellWidget(row, 2)
            note_item = table.item(row, 3)
            molecule = combo.currentData() if isinstance(combo, QComboBox) else None
            snapshot.append(
                {
                    "molecule_id": int(molecule["id"]) if isinstance(molecule, dict) else None,
                    "ratio": float(ratio_spin.value()) if isinstance(ratio_spin, QDoubleSpinBox) else 0.0,
                    "note": note_item.text().strip() if note_item is not None else "",
                }
            )
        return snapshot

    def _populate_component_table(
        self,
        table: QTableWidget,
        summary_label: QLabel,
        components: list[dict[str, Any]],
    ) -> None:
        table.setRowCount(0)
        for component in components:
            self._insert_component_row(
                table,
                molecule_id=component.get("molecule_id"),
                ratio=float(component.get("ratio", 0.0)),
                note=str(component.get("note") or ""),
            )
        self._update_ratio_summary(table, summary_label)

    def _refresh_component_table_catalog(self, table: QTableWidget, summary_label: QLabel) -> None:
        snapshot = self._snapshot_component_table(table)
        self._populate_component_table(table, summary_label, snapshot)

    def _update_ratio_summary_for_table(self, table: QTableWidget) -> None:
        if table is self.learning_component_table:
            self._update_ratio_summary(table, self.learning_ratio_summary_label)
        elif table is self.prediction_component_table:
            self._update_ratio_summary(table, self.prediction_ratio_summary_label)

    def _update_ratio_summary(self, table: QTableWidget, summary_label: QLabel) -> None:
        total = 0.0
        for row in range(table.rowCount()):
            ratio_spin = table.cellWidget(row, 2)
            if isinstance(ratio_spin, QDoubleSpinBox):
                total += float(ratio_spin.value())
        summary_label.setText(f"当前组分数: {table.rowCount()} | 比例总和: {total:.4f}")

    def _collect_components_from_table(self, table: QTableWidget) -> list[dict[str, Any]]:
        components: list[dict[str, Any]] = []
        for row in range(table.rowCount()):
            combo = table.cellWidget(row, 0)
            ratio_spin = table.cellWidget(row, 2)
            note_item = table.item(row, 3)
            molecule = combo.currentData() if isinstance(combo, QComboBox) else None
            components.append(
                {
                    "molecule_id": molecule["id"] if isinstance(molecule, dict) else None,
                    "ratio": float(ratio_spin.value()) if isinstance(ratio_spin, QDoubleSpinBox) else 0.0,
                    "note": note_item.text().strip() if note_item is not None else "",
                }
            )
        return components

    def _prepare_components_with_prompt(
        self,
        table: QTableWidget,
        *,
        action_name: str,
        summary_label: QLabel,
    ) -> dict[str, Any] | None:
        components = self._collect_components_from_table(table)
        try:
            return self.formula_service.prepare_components(components, auto_normalize=False)
        except ValueError as exc:
            message = str(exc)
            if "比例总和当前为" not in message:
                QMessageBox.warning(self, f"{action_name}失败", message)
                return None

            confirmed = QMessageBox.question(
                self,
                "比例归一化",
                f"{message}\n是否自动归一化后继续？",
            )
            if confirmed != QMessageBox.StandardButton.Yes:
                return None

        try:
            prepared = self.formula_service.prepare_components(components, auto_normalize=True)
        except ValueError as exc:
            QMessageBox.warning(self, f"{action_name}失败", str(exc))
            return None

        self._populate_component_table(table, summary_label, prepared["components"])
        return prepared

    def _normalize_component_table(self, table: QTableWidget, summary_label: QLabel) -> None:
        components = self._collect_components_from_table(table)
        try:
            prepared = self.formula_service.prepare_components(components, auto_normalize=True)
        except ValueError as exc:
            QMessageBox.warning(self, "归一化失败", str(exc))
            return
        self._populate_component_table(table, summary_label, prepared["components"])

    def _clear_learning_editor(self) -> None:
        self.learning_formula_name_input.clear()
        self.learning_formula_note_input.clear()
        self.learning_test_conditions_input.clear()
        for field_input in self.learning_target_inputs.values():
            field_input.clear()
        self.learning_component_table.setRowCount(0)
        self._update_ratio_summary(self.learning_component_table, self.learning_ratio_summary_label)

    def _refresh_saved_formulations(self) -> None:
        formulations = self.formula_service.list_formulations()
        self.saved_formulation_table.setRowCount(0)
        for row, formulation in enumerate(formulations):
            self.saved_formulation_table.insertRow(row)
            target_fields = ", ".join(sorted(formulation["target_values"].keys())) or "-"
            created_at = str(formulation["created_at"]).split("T", 1)[0]
            self.saved_formulation_table.setItem(row, 0, QTableWidgetItem(str(formulation["id"])))
            self.saved_formulation_table.setItem(row, 1, QTableWidgetItem(str(formulation["formula_name"])))
            self.saved_formulation_table.setItem(row, 2, QTableWidgetItem(target_fields))
            self.saved_formulation_table.setItem(row, 3, QTableWidgetItem(created_at))
        if formulations:
            self.saved_formulation_table.selectRow(0)
        else:
            self.saved_formulation_detail.setPlainText("暂无已保存配方。")

    def _display_selected_formulation(self) -> None:
        formulation_id = self._current_saved_formulation_id()
        if formulation_id is None:
            return
        formulation = self.formula_service.get_formulation_detail(formulation_id)
        if formulation is None:
            self.saved_formulation_detail.setPlainText("未找到该配方。")
            return

        lines = [
            f"名称: {formulation['formula_name']}",
            f"创建时间: {formulation['created_at']}",
            f"备注: {formulation['note'] or '-'}",
            "",
            "测试条件:",
        ]
        conditions = formulation["test_conditions"]
        if conditions:
            for field_name, field_value in sorted(conditions.items()):
                lines.append(f"- {field_name} = {field_value:.4f}")
        else:
            lines.append("- 暂无")

        lines.extend([
            "",
            "目标字段:",
        ])
        targets = formulation["target_values"]
        if targets:
            for field_name, field_value in sorted(targets.items()):
                lines.append(f"- {field_name} = {field_value:.4f}")
        else:
            lines.append("- 暂无")

        lines.extend(["", "组分列表:"])
        for component in formulation["components"]:
            lines.append(
                f"- {component.get('name', '-')}"
                f" | ratio={float(component.get('ratio', 0.0)):.4f}"
                f" | note={component.get('note') or '-'}"
            )

        self.saved_formulation_detail.setPlainText("\n".join(lines))

    def _current_saved_formulation_id(self) -> int | None:
        row = self.saved_formulation_table.currentRow()
        if row < 0:
            return None
        item = self.saved_formulation_table.item(row, 0)
        if item is None:
            return None
        return int(item.text())

    def _load_selected_formulation_to_editor(self) -> None:
        formulation_id = self._current_saved_formulation_id()
        if formulation_id is None:
            QMessageBox.information(self, "未选择配方", "请先在右侧列表中选择一个已保存配方。")
            return

        formulation = self.formula_service.get_formulation_detail(formulation_id)
        if formulation is None:
            QMessageBox.warning(self, "加载失败", "未找到所选配方。")
            return

        self.learning_formula_name_input.setText(str(formulation["formula_name"]))
        self.learning_formula_note_input.setPlainText(str(formulation["note"]))
        self.learning_test_conditions_input.setPlainText(
            "\n".join(
                f"{field_name.removeprefix('condition_')}={float(field_value):.4f}"
                for field_name, field_value in sorted(formulation["test_conditions"].items())
            )
        )
        for field_name, field_input in self.learning_target_inputs.items():
            value = formulation["target_values"].get(field_name)
            field_input.setText("" if value is None else f"{float(value):.4f}")
        self._populate_component_table(
            self.learning_component_table,
            self.learning_ratio_summary_label,
            list(formulation["components"]),
        )

    def _delete_selected_formulation(self) -> None:
        formulation_id = self._current_saved_formulation_id()
        if formulation_id is None:
            QMessageBox.information(self, "未选择配方", "请先选择一个已保存配方。")
            return

        confirmed = QMessageBox.question(self, "确认删除", f"确定删除配方 ID {formulation_id} 吗？")
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        self.formula_service.delete_formulation(formulation_id)
        self._refresh_saved_formulations()
        self._refresh_training_targets()

    def _save_learning_formulation(self) -> None:
        prepared = self._prepare_components_with_prompt(
            self.learning_component_table,
            action_name="保存配方",
            summary_label=self.learning_ratio_summary_label,
        )
        if prepared is None:
            return

        try:
            record_id = self.formula_service.save_formulation(
                formula_name=self.learning_formula_name_input.text(),
                note=self.learning_formula_note_input.toPlainText(),
                components=prepared["components"],
                target_values={field_name: field.text() for field_name, field in self.learning_target_inputs.items()},
                test_conditions=self.learning_test_conditions_input.toPlainText(),
                auto_normalize=False,
            )
        except (RuntimeError, ValueError) as exc:
            logger.exception("Failed to save formulation")
            QMessageBox.critical(self, "保存失败", str(exc))
            return

        QMessageBox.information(self, "保存成功", f"配方已保存，ID = {record_id}")
        self._refresh_saved_formulations()
        self._refresh_training_targets()
        self._clear_learning_editor()

    def _train_formulation_model(self) -> None:
        target_name = self.training_target_combo.currentText().strip()
        if not target_name:
            QMessageBox.information(self, "缺少目标字段", "请先选择一个训练目标字段。")
            return
        if self._training_thread is not None:
            QMessageBox.information(self, "正在训练", "当前已有配方模型训练任务正在运行。")
            return

        parameters = {
            "target_name": target_name,
            "model_key": str(self.training_model_combo.currentData()),
        }
        self._start_training_worker(parameters)

    def _start_training_worker(self, parameters: dict[str, Any]) -> None:
        self._set_training_busy(True)
        self.training_summary_label.setText(f"正在训练: {parameters['target_name']} ...")
        self.training_model_info_label.setText("训练运行中")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        thread = QThread(self)
        worker = FormulaTrainingWorker(self.formula_service, parameters)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_training_finished)
        worker.failed.connect(self._handle_training_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._handle_training_thread_finished)
        self._training_thread = thread
        self._training_worker = worker
        thread.start()

    @Slot(dict)
    def _handle_training_finished(self, artifact: dict[str, Any]) -> None:
        self.current_artifact = artifact
        self.current_explanation = None
        self.formula_shap_summary_widget.setVisible(False)
        self.formula_shap_force_widget.setVisible(False)
        self._render_training_artifact(artifact)

    @Slot(str)
    def _handle_training_failed(self, message: str) -> None:
        QMessageBox.critical(self, "训练失败", message)
        self.training_summary_label.setText("训练失败")
        self.training_model_info_label.setText("-")

    @Slot()
    def _handle_training_thread_finished(self) -> None:
        self._training_thread = None
        self._training_worker = None
        self._set_training_busy(False)
        QApplication.restoreOverrideCursor()

    def _render_training_artifact(self, artifact: dict[str, Any]) -> None:
        metrics = artifact["metrics"]
        self.training_summary_label.setText(
            f"{artifact['sample_count']} 个样本 | {len(artifact['feature_names'])} 个特征"
        )
        self.training_model_info_label.setText(f"{artifact['model_name']} | 目标: {artifact['target_name']}")
        self.training_r2_label.setText(f"{metrics['r2']:.4f}")
        self.training_mae_label.setText(f"{metrics['mae']:.4f}")
        self.training_rmse_label.setText(f"{metrics['rmse']:.4f}")
        self._update_explain_button_state()

    def _set_training_busy(self, busy: bool) -> None:
        self.training_progress.setVisible(busy)
        for widget in (
            self.training_target_combo,
            self.training_model_combo,
            self.training_train_button,
        ):
            widget.setEnabled(not busy)
        self._update_explain_button_state()

    def _update_explain_button_state(self) -> None:
        if not hasattr(self, "formula_explain_button"):
            return
        if not self.model_service.is_explainer_available():
            self.formula_explain_button.setEnabled(False)
            self.formula_explain_button.setToolTip("请安装 shap 库以启用模型解释功能")
            return
        has_artifact = self.current_artifact is not None
        can_run = has_artifact and self._training_thread is None and self._explanation_thread is None
        self.formula_explain_button.setEnabled(can_run)
        self.formula_explain_button.setToolTip("计算并展示 SHAP 全局模型解释" if has_artifact else "请先训练模型")

    def _explain_model(self) -> None:
        if self.current_artifact is None:
            QMessageBox.information(self, "未训练模型", "请先完成一次模型训练。")
            return
        if not self.model_service.is_explainer_available():
            QMessageBox.information(self, "缺少 SHAP", "请安装 shap 库以启用模型解释功能。")
            return
        if self._explanation_thread is not None:
            QMessageBox.information(self, "正在解释", "当前已有模型解释任务正在运行。")
            return

        self.formula_explain_button.setText("解释计算中...")
        self.formula_explain_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        thread = QThread(self)
        worker = FormulaExplanationWorker(self.model_service, dict(self.current_artifact))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._handle_explanation_finished)
        worker.failed.connect(self._handle_explanation_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._handle_explanation_thread_finished)
        self._explanation_thread = thread
        self._explanation_worker = worker
        thread.start()

    @Slot(object)
    def _handle_explanation_finished(self, explanation: object) -> None:
        self.current_explanation = explanation
        self.formula_shap_summary_widget.load_explanation(explanation)  # type: ignore[arg-type]
        self.formula_shap_summary_widget.setVisible(True)

    @Slot(str)
    def _handle_explanation_failed(self, message: str) -> None:
        QMessageBox.warning(self, "模型解释失败", message)

    @Slot()
    def _handle_explanation_thread_finished(self) -> None:
        self._explanation_thread = None
        self._explanation_worker = None
        self.formula_explain_button.setText("模型解释")
        self._update_explain_button_state()
        QApplication.restoreOverrideCursor()

    def _predict_new_formulation(self) -> None:
        if self.current_artifact is None:
            QMessageBox.information(self, "未训练模型", "请先在左侧完成一次模型训练。")
            return

        prepared = self._prepare_components_with_prompt(
            self.prediction_component_table,
            action_name="预测配方",
            summary_label=self.prediction_ratio_summary_label,
        )
        if prepared is None:
            return

        try:
            prediction = self.formula_service.predict_formulation(
                self.current_artifact,
                prepared["components"],
                test_conditions=self.prediction_test_conditions_input.toPlainText(),
                auto_normalize=False,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            logger.exception("Failed to predict formulation")
            QMessageBox.critical(self, "预测失败", str(exc))
            return

        lines = [
            f"目标字段: {prediction['target_name']}",
            f"模型: {prediction['model_name']}",
            f"预测值: {prediction['prediction']:.4f}",
            "",
            "组分明细:",
        ]
        for component in prediction["components"]:
            lines.append(f"- {component['name']} | ratio={float(component['ratio']):.4f}")

        lines.extend(["", "测试条件:"])
        if prediction["test_conditions"]:
            for field_name, field_value in sorted(prediction["test_conditions"].items()):
                lines.append(f"- {field_name} = {float(field_value):.4f}")
        else:
            lines.append("- 暂无")

        lines.extend(["", "特征预览:"])
        for feature_name, feature_value in list(prediction["features"].items())[:10]:
            lines.append(f"- {feature_name} = {float(feature_value):.4f}")

        self.prediction_result_text.setPlainText("\n".join(lines))
        self._render_single_prediction_explanation(prediction["features"])

    def _render_single_prediction_explanation(self, feature_values: dict[str, float]) -> None:
        if self.current_artifact is None or self.current_explanation is None:
            self.formula_shap_force_widget.setVisible(False)
            return
        try:
            feature_frame = pd.DataFrame(
                [
                    {
                        str(feature): float(feature_values.get(str(feature), 0.0))
                        for feature in list(self.current_artifact["feature_names"])
                    }
                ]
            )
            payload = self.model_service.explain_single_prediction(
                self.current_artifact,
                self.current_explanation,
                feature_frame,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Failed to render formulation local SHAP explanation: %s", exc)
            self.formula_shap_force_widget.setVisible(False)
            return
        self.formula_shap_force_widget.load_force_plot(str(payload["html"]))
        self.formula_shap_force_widget.setVisible(True)
