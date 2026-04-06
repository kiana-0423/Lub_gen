from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QComboBox,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.services.feature_service import FeatureService
from chemstudio.services.model_service import ModelService
from chemstudio.services.visualization_service import VisualizationService
from chemstudio.ui.widgets import BasePage, MatplotlibCanvas
from chemstudio.utils.config import AppConfig


class MoleculeDesignPage(BasePage):
    """Regression model training and single-molecule prediction page."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        feature_service: FeatureService,
        model_service: ModelService,
        visualization_service: VisualizationService,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.feature_service = feature_service
        self.model_service = model_service
        self.visualization_service = visualization_service
        self.current_artifact: dict[str, object] | None = None
        self._build_ui()
        self.refresh_page()

    def _build_ui(self) -> None:
        root_layout = self.create_page_shell(
            "分子设计",
            "从数据库加载训练数据，训练回归模型，并对单个分子输入进行性能预测。",
        )

        content_layout = QHBoxLayout()
        content_layout.addWidget(self._build_training_panel(), stretch=3)
        content_layout.addWidget(self._build_prediction_panel(), stretch=2)
        root_layout.addLayout(content_layout, stretch=1)

    def _build_training_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        controls_box = QGroupBox("模型训练")
        controls_form = QGridLayout(controls_box)

        self.target_combo = QComboBox()
        self.model_combo = QComboBox()
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setDecimals(2)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setRange(0.1, 0.5)
        self.test_size_spin.setValue(0.2)

        refresh_button = QPushButton("刷新训练数据")
        refresh_button.clicked.connect(self.refresh_page)
        train_button = QPushButton("开始训练")
        train_button.clicked.connect(self._train_model)
        save_button = QPushButton("保存模型")
        save_button.clicked.connect(self._save_model)
        load_button = QPushButton("读取模型")
        load_button.clicked.connect(self._load_model)

        controls_form.addWidget(QLabel("目标性能"), 0, 0)
        controls_form.addWidget(self.target_combo, 0, 1)
        controls_form.addWidget(QLabel("模型"), 1, 0)
        controls_form.addWidget(self.model_combo, 1, 1)
        controls_form.addWidget(QLabel("测试集比例"), 2, 0)
        controls_form.addWidget(self.test_size_spin, 2, 1)
        controls_form.addWidget(refresh_button, 3, 0)
        controls_form.addWidget(train_button, 3, 1)
        controls_form.addWidget(save_button, 4, 0)
        controls_form.addWidget(load_button, 4, 1)

        metrics_box = QGroupBox("训练结果")
        metrics_form = QFormLayout(metrics_box)
        self.dataset_info_label = QLabel("-")
        self.metric_r2_label = QLabel("-")
        self.metric_mae_label = QLabel("-")
        self.metric_rmse_label = QLabel("-")
        self.model_info_label = QLabel("-")
        metrics_form.addRow("数据概况", self.dataset_info_label)
        metrics_form.addRow("模型信息", self.model_info_label)
        metrics_form.addRow("R²", self.metric_r2_label)
        metrics_form.addRow("MAE", self.metric_mae_label)
        metrics_form.addRow("RMSE", self.metric_rmse_label)

        self.canvas = MatplotlibCanvas(width=6.4, height=4.6)

        layout.addWidget(controls_box)
        layout.addWidget(metrics_box)
        layout.addWidget(self.canvas, stretch=1)
        return panel

    def _build_prediction_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        prediction_box = QGroupBox("单分子预测")
        prediction_layout = QVBoxLayout(prediction_box)

        self.smiles_input = QLineEdit()
        self.smiles_input.setPlaceholderText("输入 SMILES，例如 CCO")

        self.feature_text_edit = QPlainTextEdit()
        self.feature_text_edit.setPlaceholderText(
            '输入 JSON 或 key=value 特征，例如:\n{"mol_wt": 88.0, "tpsa": 26.3}\n或\nmol_wt=88.0\ntpsa=26.3'
        )

        predict_button = QPushButton("预测性能")
        predict_button.clicked.connect(self._predict_single_molecule)

        self.prediction_result_label = QLabel("尚未预测")
        self.prediction_result_label.setWordWrap(True)

        prediction_layout.addWidget(QLabel("SMILES"))
        prediction_layout.addWidget(self.smiles_input)
        prediction_layout.addWidget(QLabel("手动特征输入"))
        prediction_layout.addWidget(self.feature_text_edit, stretch=1)
        prediction_layout.addWidget(predict_button)
        prediction_layout.addWidget(self.prediction_result_label)

        layout.addWidget(prediction_box, stretch=1)
        return panel

    def refresh_page(self) -> None:
        dataset = self.model_service.get_training_dataset()
        property_names = self.model_service.get_target_columns()
        self.dataset_info_label.setText(
            f"{len(dataset)} 行, {len(dataset.columns)} 列, {len(self.db_manager.list_feature_names())} 个特征候选"
        )

        self.target_combo.clear()
        for property_name in property_names:
            self.target_combo.addItem(property_name)

        self.model_combo.clear()
        for item in self.model_service.get_model_catalog():
            label = str(item["label"])
            if not item["available"]:
                label = f"{label} (未安装)"
            self.model_combo.addItem(label, userData=item["key"])

    def _train_model(self) -> None:
        if self.target_combo.count() == 0:
            QMessageBox.warning(self, "缺少目标列", "数据库中没有可训练的性能标签。")
            return

        target_name = self.target_combo.currentText()
        model_key = str(self.model_combo.currentData())
        try:
            artifact = self.model_service.train_model(
                target_name=target_name,
                model_key=model_key,
                test_size=float(self.test_size_spin.value()),
            )
        except Exception as exc:
            QMessageBox.critical(self, "训练失败", str(exc))
            return

        self.current_artifact = artifact
        metrics = artifact["metrics"]
        self.metric_r2_label.setText(f"{metrics['r2']:.4f}")
        self.metric_mae_label.setText(f"{metrics['mae']:.4f}")
        self.metric_rmse_label.setText(f"{metrics['rmse']:.4f}")
        self.model_info_label.setText(
            f"{artifact['model_name']} | 目标: {artifact['target_name']} | 特征数: {len(artifact['feature_names'])}"
        )
        self.visualization_service.plot_prediction_scatter(
            self.canvas.axes,
            list(artifact["y_true"]),
            list(artifact["y_pred"]),
        )
        self.canvas.draw_idle()

    def _save_model(self) -> None:
        if self.current_artifact is None:
            QMessageBox.information(self, "无可保存模型", "请先训练模型或加载已保存模型。")
            return
        default_name = (
            f"{self.current_artifact['target_name']}_{self.current_artifact['model_key']}.joblib"
            if self.current_artifact
            else "chemstudio_model.joblib"
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存模型",
            str(AppConfig.SAVED_MODELS_DIR / default_name),
            "Model Files (*.joblib)",
        )
        if not file_path:
            return
        self.model_service.save_model(self.current_artifact, file_path)
        QMessageBox.information(self, "保存完成", f"模型已保存到:\n{Path(file_path).name}")

    def _load_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "读取模型",
            str(AppConfig.SAVED_MODELS_DIR),
            "Model Files (*.joblib)",
        )
        if not file_path:
            return
        try:
            artifact = self.model_service.load_model(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "读取失败", str(exc))
            return

        self.current_artifact = artifact
        self.model_info_label.setText(
            f"{artifact['model_name']} | 目标: {artifact['target_name']} | 特征数: {len(artifact['feature_names'])}"
        )
        metrics = artifact.get("metrics") or {}
        if metrics:
            self.metric_r2_label.setText(f"{metrics.get('r2', 0.0):.4f}")
            self.metric_mae_label.setText(f"{metrics.get('mae', 0.0):.4f}")
            self.metric_rmse_label.setText(f"{metrics.get('rmse', 0.0):.4f}")
        y_true = list(artifact.get("y_true") or [])
        y_pred = list(artifact.get("y_pred") or [])
        if y_true and y_pred:
            self.visualization_service.plot_prediction_scatter(self.canvas.axes, y_true, y_pred)
            self.canvas.draw_idle()

    def _predict_single_molecule(self) -> None:
        if self.current_artifact is None:
            QMessageBox.information(self, "未加载模型", "请先训练模型或读取已保存模型。")
            return

        try:
            feature_values, report = self.feature_service.build_prediction_features(
                smiles=self.smiles_input.text(),
                feature_text=self.feature_text_edit.toPlainText(),
                required_features=list(self.current_artifact["feature_names"]),
            )
            prediction = self.model_service.predict(self.current_artifact, feature_values)
        except Exception as exc:
            QMessageBox.critical(self, "预测失败", str(exc))
            return

        self.prediction_result_label.setText(
            "\n".join(
                [
                    f"目标性能: {self.current_artifact['target_name']}",
                    f"预测值: {prediction:.4f}",
                    f"特征覆盖: {len(report['merged_features'])} / {len(self.current_artifact['feature_names'])}",
                    f"提示: {report['message']}",
                ]
            )
        )
