from __future__ import annotations

from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.services.formula_service import FormulaService
from chemstudio.services.model_service import ModelService
from chemstudio.ui.widgets import BasePage
from chemstudio.utils.config import AppConfig


class FormulaDesignPage(BasePage):
    """Formula composition builder and formula-property prediction page."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        formula_service: FormulaService,
        model_service: ModelService,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.formula_service = formula_service
        self.model_service = model_service
        self.current_artifact: dict[str, object] | None = None
        self.last_prediction: dict[str, object] | None = None
        self._build_ui()
        self.refresh_page()

    def _build_ui(self) -> None:
        root_layout = self.create_page_shell(
            "配方设计",
            "选择多个分子组分并设置配比，采用加权平均特征工程生成配方特征，并调用训练好的模型进行预测。",
        )

        top_controls = QHBoxLayout()
        self.formula_name_input = QLineEdit()
        self.formula_name_input.setPlaceholderText("输入配方名称")
        self.model_info_label = QLabel("尚未加载模型")

        load_model_button = QPushButton("加载模型")
        load_model_button.clicked.connect(self._load_model)
        clear_button = QPushButton("清空组分")
        clear_button.clicked.connect(self._clear_components)

        top_controls.addWidget(QLabel("配方名称"))
        top_controls.addWidget(self.formula_name_input, stretch=1)
        top_controls.addWidget(load_model_button)
        top_controls.addWidget(clear_button)
        top_controls.addWidget(self.model_info_label, stretch=1)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self._build_component_panel(), stretch=3)
        content_layout.addWidget(self._build_result_panel(), stretch=2)

        root_layout.addLayout(top_controls)
        root_layout.addLayout(content_layout, stretch=1)

    def _build_component_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        selector_box = QGroupBox("配方组分")
        selector_layout = QGridLayout(selector_box)
        self.molecule_combo = QComboBox()
        add_button = QPushButton("添加组分")
        add_button.clicked.connect(self._add_component)
        remove_button = QPushButton("移除选中")
        remove_button.clicked.connect(self._remove_selected_component)

        selector_layout.addWidget(QLabel("分子"), 0, 0)
        selector_layout.addWidget(self.molecule_combo, 0, 1)
        selector_layout.addWidget(add_button, 0, 2)
        selector_layout.addWidget(remove_button, 0, 3)

        self.component_table = QTableWidget(0, 4)
        self.component_table.setHorizontalHeaderLabels(["ID", "名称", "SMILES", "比例"])
        self.component_table.horizontalHeader().setStretchLastSection(True)

        action_layout = QHBoxLayout()
        predict_button = QPushButton("预测配方性能")
        predict_button.clicked.connect(self._predict_formula)
        save_button = QPushButton("保存配方记录")
        save_button.clicked.connect(self._save_formula)
        action_layout.addWidget(predict_button)
        action_layout.addWidget(save_button)
        action_layout.addStretch(1)

        layout.addWidget(selector_box)
        layout.addWidget(self.component_table, stretch=1)
        layout.addLayout(action_layout)
        return panel

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        result_box = QGroupBox("预测结果")
        result_layout = QVBoxLayout(result_box)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)

        layout.addWidget(result_box, stretch=1)
        return panel

    def refresh_page(self) -> None:
        self.molecule_combo.clear()
        for molecule in self.db_manager.list_molecules():
            label = f"{molecule['id']} | {molecule['name']}"
            self.molecule_combo.addItem(label, userData=molecule)

    def _load_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载模型",
            str(AppConfig.SAVED_MODELS_DIR),
            "Model Files (*.joblib)",
        )
        if not file_path:
            return
        try:
            self.current_artifact = self.model_service.load_model(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "读取模型失败", str(exc))
            return

        self.model_info_label.setText(
            f"{self.current_artifact['model_name']} -> {self.current_artifact['target_name']}"
        )

    def _add_component(self) -> None:
        molecule = self.molecule_combo.currentData()
        if molecule is None:
            return

        existing_ids = {self.component_table.item(row, 0).text() for row in range(self.component_table.rowCount())}
        molecule_id = str(molecule["id"])
        if molecule_id in existing_ids:
            QMessageBox.information(self, "重复组分", "该分子已经在当前配方中。")
            return

        row = self.component_table.rowCount()
        self.component_table.insertRow(row)
        self.component_table.setItem(row, 0, QTableWidgetItem(str(molecule["id"])))
        self.component_table.setItem(row, 1, QTableWidgetItem(str(molecule["name"])))
        self.component_table.setItem(row, 2, QTableWidgetItem(str(molecule["smiles"] or "")))
        self.component_table.setItem(row, 3, QTableWidgetItem("0.0"))

    def _remove_selected_component(self) -> None:
        row = self.component_table.currentRow()
        if row >= 0:
            self.component_table.removeRow(row)

    def _clear_components(self) -> None:
        self.component_table.setRowCount(0)
        self.result_text.clear()
        self.last_prediction = None

    def _collect_components(self) -> list[dict[str, float | int | str]]:
        components: list[dict[str, float | int | str]] = []
        for row in range(self.component_table.rowCount()):
            molecule_id = int(self.component_table.item(row, 0).text())
            ratio_text = self.component_table.item(row, 3).text().strip()
            components.append({"molecule_id": molecule_id, "ratio": float(ratio_text)})
        return components

    def _predict_formula(self) -> None:
        if self.current_artifact is None:
            QMessageBox.information(self, "未加载模型", "请先加载训练好的模型。")
            return

        try:
            components = self._collect_components()
            prediction = self.formula_service.predict_formula(self.current_artifact, components)
        except Exception as exc:
            QMessageBox.critical(self, "预测失败", str(exc))
            return

        self.last_prediction = prediction
        lines = [
            f"目标性能: {prediction['target_name']}",
            f"预测值: {prediction['prediction']:.4f}",
            "",
            "组分明细:",
        ]
        for component in prediction["components"]:
            lines.append(f"- {component['name']} | ratio={component['ratio']:.4f}")

        lines.extend(["", "配方特征预览:"])
        for feature_name, feature_value in list(prediction["features"].items())[:10]:
            lines.append(f"- {feature_name} = {feature_value:.4f}")

        self.result_text.setPlainText("\n".join(lines))

    def _save_formula(self) -> None:
        if self.last_prediction is None:
            QMessageBox.information(self, "无预测结果", "请先完成一次配方预测。")
            return

        formula_name = self.formula_name_input.text().strip() or "unnamed_formula"
        try:
            record_id = self.formula_service.save_formula_result(formula_name, self.last_prediction)
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return

        QMessageBox.information(self, "保存成功", f"配方记录已保存，ID = {record_id}")
