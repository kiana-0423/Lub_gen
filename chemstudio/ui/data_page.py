from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - depends on optional Qt WebEngine runtime
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover
    QWebEngineView = None

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.repositories import MoleculeRepository
from chemstudio.ml.featurizers import is_mordred_available
from chemstudio.services.data_import_service import DataImportService
from chemstudio.services.visualization_service import VisualizationService
from chemstudio.ui.widgets import BasePage, PandasTableModel


logger = logging.getLogger(__name__)


class DataPage(BasePage):
    """Data import, browsing, filtering, and molecule visualization page."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        data_import_service: DataImportService,
        visualization_service: VisualizationService,
        molecule_repository: MoleculeRepository | None = None,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
        self.molecule_repository = molecule_repository or MoleculeRepository(db_manager)
        self.data_import_service = data_import_service
        self.visualization_service = visualization_service
        self.dataset_model = PandasTableModel()
        self.dataset = self.dataset_model.dataframe
        self._build_ui()
        self.refresh_page()

    def _build_ui(self) -> None:
        root_layout = self.create_page_shell(
            "数据导入与可视化",
            "导入 CSV/Excel，保存至 SQLite，浏览数据，并查看当前选中分子的 3D 结构。",
        )

        control_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("按名称、SMILES 或来源检索")
        self.material_type_filter = QComboBox()
        self.material_type_filter.addItem("全部", None)
        self.material_type_filter.addItem("基础油", "base_oil")
        self.material_type_filter.addItem("添加剂", "additive")
        self.material_type_filter.currentIndexChanged.connect(self.refresh_page)

        import_button = QPushButton("导入文件")
        import_button.clicked.connect(self._import_file)

        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh_page)

        self.compute_descriptors_button = QPushButton("计算描述符")
        self.compute_descriptors_button.clicked.connect(self._compute_descriptors_for_current_rows)
        if is_mordred_available():
            self.compute_descriptors_button.setToolTip("为当前筛选结果中缺失描述符的分子计算 Mordred 描述符")
        else:
            self.compute_descriptors_button.setEnabled(False)
            self.compute_descriptors_button.setToolTip("Mordred 未安装，无法计算描述符")

        self.export_features_button = QPushButton("导出特征 CSV")
        self.export_features_button.clicked.connect(self._export_features_csv)
        if is_mordred_available():
            self.export_features_button.setToolTip("导出当前筛选后的 Mordred 特征宽表")
        else:
            self.export_features_button.setEnabled(False)
            self.export_features_button.setToolTip("Mordred 未安装，无法导出描述符特征")

        delete_button = QPushButton("删除选中分子")
        delete_button.clicked.connect(self._delete_selected_molecule)

        control_layout.addWidget(QLabel("检索"))
        control_layout.addWidget(self.search_input, stretch=1)
        control_layout.addWidget(QLabel("材料类型"))
        control_layout.addWidget(self.material_type_filter)
        control_layout.addWidget(import_button)
        control_layout.addWidget(refresh_button)
        control_layout.addWidget(self.compute_descriptors_button)
        control_layout.addWidget(self.export_features_button)
        control_layout.addWidget(delete_button)

        splitter = QSplitter()
        splitter.addWidget(self._build_table_panel())
        splitter.addWidget(self._build_viewer_panel())
        splitter.setSizes([900, 520])

        self.status_label = QLabel()
        root_layout.addLayout(control_layout)
        root_layout.addWidget(splitter, stretch=1)
        root_layout.addWidget(self.status_label)

    def _build_table_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table_view = QTableView()
        self.table_view.setModel(self.dataset_model)
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(False)
        self.table_view.selectionModel().currentRowChanged.connect(self._handle_current_row_changed)

        layout.addWidget(self.table_view)
        return panel

    def _build_viewer_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        viewer_box = QGroupBox("分子 3D 可视化")
        viewer_layout = QVBoxLayout(viewer_box)
        viewer_layout.setSpacing(10)

        controls_layout = QHBoxLayout()
        self.reset_view_button = QPushButton("重置视角")
        self.reset_view_button.clicked.connect(self._reset_view)
        reload_button = QPushButton("重新加载")
        reload_button.clicked.connect(self._reload_selected_molecule)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.reset_view_button)
        controls_layout.addWidget(reload_button)

        if QWebEngineView is None:
            self.molecule_view = None
            self.viewer_fallback_label = QLabel(
                "当前环境缺少 Qt WebEngine，无法显示分子 3D 视图。\n请安装 PySide6-Addons 后重试。"
            )
            self.viewer_fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.viewer_fallback_label.setWordWrap(True)
            self.viewer_fallback_label.setMinimumHeight(420)
            self.reset_view_button.setEnabled(False)
            viewer_layout.addLayout(controls_layout)
            viewer_layout.addWidget(self.viewer_fallback_label, stretch=1)
        else:
            self.molecule_view = QWebEngineView()
            self.molecule_view.setMinimumHeight(420)
            viewer_layout.addLayout(controls_layout)
            viewer_layout.addWidget(self.molecule_view, stretch=1)

        layout.addWidget(viewer_box, stretch=1)
        layout.addWidget(self._build_compatibility_panel())
        return panel

    def _build_compatibility_panel(self) -> QWidget:
        self.compatibility_box = QGroupBox("基础油相容性")
        self.compatibility_box.setVisible(False)
        layout = QVBoxLayout(self.compatibility_box)
        layout.setSpacing(8)

        self.compatibility_table = QTableWidget(0, 5)
        self.compatibility_table.setHorizontalHeaderLabels(["基础油", "评分", "溶解度", "备注", "操作"])
        self.compatibility_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.compatibility_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.compatibility_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.compatibility_table.verticalHeader().setVisible(False)
        self.compatibility_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.compatibility_table)
        return self.compatibility_box

    def refresh_page(self) -> None:
        self.dataset = self._build_current_dataset(include_mordred=False)
        self.dataset_model.set_dataframe(self.dataset)
        self.table_view.resizeColumnsToContents()
        descriptor_molecule_count = self.molecule_repository.count_descriptor_rows()
        self.status_label.setText(
            f"当前记录数: {len(self.dataset)} | "
            f"数值列: {len(self.dataset.select_dtypes(include='number').columns)} | "
            f"描述符已计算: {descriptor_molecule_count} 个分子"
        )
        self._sync_table_selection()

    def _selected_material_type(self) -> str | None:
        if not hasattr(self, "material_type_filter"):
            return None
        value = self.material_type_filter.currentData()
        return str(value) if value else None

    def _build_current_dataset(self, *, include_mordred: bool, search_text: str | None = None) -> object:
        if search_text is None:
            search_text = self.search_input.text().strip() if hasattr(self, "search_input") else ""
        dataset = self.molecule_repository.get_wide_dataset(search_text=search_text, include_mordred=include_mordred)
        material_type = self._selected_material_type()
        if material_type is None or dataset.empty:
            return dataset
        material_type_ids = [int(row["id"]) for row in self.db_manager.list_material_types(material_type)]
        if not material_type_ids:
            return dataset.iloc[0:0].copy()
        if "material_type_id" in dataset.columns:
            return dataset[dataset["material_type_id"].isin(material_type_ids)].reset_index(drop=True)
        matching_ids: set[int] = set()
        for material_type_id in material_type_ids:
            matching_rows = self.db_manager.list_molecules(
                search_text=search_text,
                material_type_id=material_type_id,
            )
            matching_ids.update(int(row["id"]) for row in matching_rows)
        return dataset[dataset["id"].isin(matching_ids)].reset_index(drop=True)

    def _import_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入数据文件",
            "",
            "Tabular Files (*.csv *.xlsx *.xls)",
        )
        if not file_path:
            return
        try:
            result = self.data_import_service.import_file(file_path)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Failed to import data file: %s", file_path)
            QMessageBox.critical(self, "导入失败", str(exc))
            return

        self.refresh_page()
        QMessageBox.information(
            self,
            "导入完成",
            f"已导入 {result['row_count']} 条记录。\n来源文件: {Path(file_path).name}",
        )

    def _compute_descriptors_for_current_rows(self) -> None:
        if not is_mordred_available():
            QMessageBox.information(self, "无法计算", "Mordred 未安装，无法计算描述符。")
            return
        if self.dataset.empty:
            QMessageBox.information(self, "没有数据", "当前没有可计算描述符的分子记录。")
            return

        molecule_ids = [int(value) for value in self.dataset["id"].dropna().tolist()]
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = self.data_import_service.compute_missing_descriptors(molecule_ids)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Failed to compute descriptors for current rows.")
            QMessageBox.critical(self, "描述符计算失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.refresh_page()
        QMessageBox.information(
            self,
            "描述符计算完成",
            "已处理当前筛选结果。\n"
            f"新计算: {result['computed_count']} 个分子\n"
            f"已跳过: {result['skipped_count']} 个分子\n"
            f"失败/无有效描述符: {result['failed_count']} 个分子",
        )

    def _export_features_csv(self) -> None:
        if not is_mordred_available():
            QMessageBox.information(self, "无法导出", "Mordred 未安装，无法导出描述符特征。")
            return

        default_path = self._default_feature_export_path()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 Mordred 特征 CSV",
            str(default_path),
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        destination = Path(file_path)
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            row_count, column_count = self._write_features_csv(destination)
            if row_count == 0 or column_count == 0:
                QMessageBox.information(
                    self,
                    "没有可导出的特征",
                    "当前没有可导出的特征数据，请先导入分子并计算描述符。",
                )
                return
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Failed to export Mordred feature CSV: %s", destination)
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.status_label.setText(
            f"已导出 {row_count} 行 × {column_count} 列特征数据到 {destination}"
        )

    def _write_features_csv(self, destination: Path, search_text: str | None = None) -> tuple[int, int]:
        """Write the current filtered Mordred wide table and return its shape."""
        export_frame = self._build_current_dataset(include_mordred=True, search_text=search_text)
        if export_frame.empty or self.molecule_repository.count_descriptor_rows() == 0:
            return 0, 0
        export_frame.to_csv(destination, index=False, encoding="utf-8-sig")
        return int(len(export_frame)), int(len(export_frame.columns))

    def _default_feature_export_path(self) -> Path:
        export_dir = Path.home() / "Desktop"
        if not export_dir.exists():
            export_dir = Path.home()
        date_label = datetime.now().strftime("%Y-%m-%d")
        return export_dir / f"mordred_features_{date_label}.csv"

    def _delete_selected_molecule(self) -> None:
        if self.dataset.empty:
            return
        index = self.table_view.currentIndex()
        if not index.isValid():
            QMessageBox.information(self, "未选择记录", "请先在表格中选择一条分子记录。")
            return

        molecule_id = int(self.dataset.iloc[index.row()]["id"])
        confirmed = QMessageBox.question(self, "确认删除", f"确定删除分子 ID {molecule_id} 吗？")
        if confirmed != QMessageBox.StandardButton.Yes:
            return

        if not self.molecule_repository.delete_molecule(molecule_id):
            QMessageBox.warning(self, "删除失败", f"未找到分子 ID {molecule_id}，请刷新后重试。")
            return
        self.refresh_page()
        QMessageBox.information(self, "删除完成", f"已删除分子 ID {molecule_id}。")

    def _sync_table_selection(self) -> None:
        if self.dataset.empty:
            self.table_view.clearSelection()
            self._render_selected_molecule(error_message="暂无可显示的分子数据。")
            self._render_compatibility_panel(None)
            return

        index = self.table_view.currentIndex()
        if not index.isValid() or index.row() >= len(self.dataset):
            self.table_view.selectRow(0)
            return

        self._render_selected_molecule()

    def _handle_current_row_changed(self, current, previous) -> None:
        del current, previous
        self._render_selected_molecule()

    def _reload_selected_molecule(self) -> None:
        self._render_selected_molecule()

    def _reset_view(self) -> None:
        if self.molecule_view is None:
            return
        self.molecule_view.page().runJavaScript("window.resetView && window.resetView();")

    def _render_selected_molecule(self, error_message: str | None = None) -> None:
        if error_message is not None:
            self._set_molecule_view(
                molecule_name="分子 3D 可视化",
                smiles="",
                molblock=None,
                error_message=error_message,
            )
            self._render_compatibility_panel(None)
            return

        index = self.table_view.currentIndex()
        if self.dataset.empty or not index.isValid() or index.row() >= len(self.dataset):
            self._set_molecule_view(
                molecule_name="分子 3D 可视化",
                smiles="",
                molblock=None,
                error_message="请先在左侧表格中选择一条分子记录。",
            )
            return

        row = self.dataset.iloc[index.row()]
        molecule_name = str(row.get("name") or f"分子 ID {row.get('id', '-')}")
        smiles = str(row.get("smiles") or "").strip()
        molblock, generation_error = self.visualization_service.generate_3d_molblock(smiles)
        self._set_molecule_view(
            molecule_name=molecule_name,
            smiles=smiles,
            molblock=molblock,
            error_message=generation_error,
        )
        self._render_compatibility_panel(int(row["id"]) if row.get("id") is not None else None)

    def _set_molecule_view(
        self,
        *,
        molecule_name: str,
        smiles: str,
        molblock: str | None,
        error_message: str | None,
    ) -> None:
        if self.molecule_view is None:
            fallback_lines = ["当前环境缺少 Qt WebEngine，无法显示分子 3D 视图。", "请安装 PySide6-Addons 后重试。"]
            if error_message:
                fallback_lines.append(error_message)
            self.viewer_fallback_label.setText(
                f"{molecule_name}\nSMILES: {smiles or '-'}\n\n" + "\n".join(fallback_lines)
            )
            return

        html = self.visualization_service.build_molecule_viewer_html(
            molblock=molblock,
            molecule_name=molecule_name,
            smiles=smiles,
            error_message=error_message,
        )
        self.molecule_view.setHtml(html, QUrl("https://3dmol.org/"))

    def _render_compatibility_panel(self, molecule_id: int | None) -> None:
        if molecule_id is None or not self._is_additive_molecule(molecule_id):
            self.compatibility_box.setVisible(False)
            self.compatibility_table.setRowCount(0)
            return

        base_oils = self._list_base_oil_molecules()
        compatibility_by_base = {
            int(row["base_oil_id"]): row
            for row in self.db_manager.get_additive_compatibilities(additive_id=molecule_id)
        }
        self.compatibility_table.setRowCount(0)
        for row_index, base_oil in enumerate(base_oils):
            base_oil_id = int(base_oil["id"])
            compatibility = compatibility_by_base.get(base_oil_id, {})
            self.compatibility_table.insertRow(row_index)
            self.compatibility_table.setItem(row_index, 0, QTableWidgetItem(str(base_oil.get("name") or base_oil_id)))
            score = compatibility.get("compatibility_score")
            self.compatibility_table.setItem(
                row_index,
                1,
                QTableWidgetItem("-" if score is None else f"{float(score):.3f}"),
            )
            self.compatibility_table.setItem(row_index, 2, QTableWidgetItem(str(compatibility.get("solubility") or "")))
            self.compatibility_table.setItem(row_index, 3, QTableWidgetItem(str(compatibility.get("notes") or "")))
            edit_button = QPushButton("编辑")
            self._connect_compatibility_edit_button(
                edit_button,
                additive_id=molecule_id,
                base_oil_id=base_oil_id,
                base_oil_name=str(base_oil.get("name") or base_oil_id),
                compatibility=dict(compatibility),
            )
            self.compatibility_table.setCellWidget(row_index, 4, edit_button)
        if not base_oils:
            self.compatibility_table.setRowCount(1)
            self.compatibility_table.setSpan(0, 0, 1, 5)
            self.compatibility_table.setItem(0, 0, QTableWidgetItem("暂无基础油记录。"))
        self.compatibility_box.setVisible(True)
        self.compatibility_table.resizeColumnsToContents()

    def _connect_compatibility_edit_button(
        self,
        button: QPushButton,
        *,
        additive_id: int,
        base_oil_id: int,
        base_oil_name: str,
        compatibility: dict[str, object],
    ) -> None:
        button.clicked.connect(
            lambda _checked=False: self._edit_compatibility(
                additive_id,
                base_oil_id,
                base_oil_name,
                compatibility,
            )
        )

    def _edit_compatibility(
        self,
        additive_id: int,
        base_oil_id: int,
        base_oil_name: str,
        compatibility: dict[str, object],
    ) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"编辑相容性 - {base_oil_name}")
        layout = QFormLayout(dialog)

        score_input = QDoubleSpinBox()
        score_input.setDecimals(3)
        score_input.setRange(0.0, 1.0)
        score_input.setSingleStep(0.05)
        if compatibility.get("compatibility_score") is not None:
            score_input.setValue(float(compatibility["compatibility_score"]))

        solubility_input = QComboBox()
        for value in ["", "soluble", "partially_soluble", "insoluble"]:
            solubility_input.addItem(value or "-", value)
        solubility_index = solubility_input.findData(str(compatibility.get("solubility") or ""))
        solubility_input.setCurrentIndex(solubility_index if solubility_index >= 0 else 0)

        notes_input = QTextEdit()
        notes_input.setMaximumHeight(96)
        notes_input.setPlainText(str(compatibility.get("notes") or ""))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addRow("基础油", QLabel(base_oil_name))
        layout.addRow("评分", score_input)
        layout.addRow("溶解度", solubility_input)
        layout.addRow("备注", notes_input)
        layout.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        self.db_manager.save_additive_compatibility(
            additive_id=additive_id,
            base_oil_id=base_oil_id,
            compatibility_score=float(score_input.value()),
            solubility=str(solubility_input.currentData() or ""),
            notes=notes_input.toPlainText().strip(),
        )
        self._render_compatibility_panel(additive_id)

    def _is_additive_molecule(self, molecule_id: int) -> bool:
        detail = self.db_manager.get_molecule_detail(molecule_id)
        if detail is None or detail.material_type_id is None:
            return False
        material_type = self._material_type_by_id(detail.material_type_id)
        return material_type is not None and str(material_type.get("type_name")) == "additive"

    def _list_base_oil_molecules(self) -> list[dict[str, object]]:
        base_oil_type_ids = [int(row["id"]) for row in self.db_manager.list_material_types("base_oil")]
        base_oils: list[dict[str, object]] = []
        for material_type_id in base_oil_type_ids:
            base_oils.extend(
                self.db_manager.list_molecules(
                    include_hidden=True,
                    material_type_id=material_type_id,
                    sort_by="name",
                    descending=False,
                )
            )
        return base_oils

    def _material_type_by_id(self, material_type_id: int) -> dict[str, object] | None:
        for row in self.db_manager.list_material_types():
            if int(row["id"]) == int(material_type_id):
                return row
        return None
