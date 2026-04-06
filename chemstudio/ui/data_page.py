from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

try:  # pragma: no cover - depends on optional Qt WebEngine runtime
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover
    QWebEngineView = None

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.services.data_import_service import DataImportService
from chemstudio.services.visualization_service import VisualizationService
from chemstudio.ui.widgets import BasePage, PandasTableModel


class DataPage(BasePage):
    """Data import, browsing, filtering, and molecule visualization page."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        data_import_service: DataImportService,
        visualization_service: VisualizationService,
    ) -> None:
        super().__init__()
        self.db_manager = db_manager
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

        import_button = QPushButton("导入文件")
        import_button.clicked.connect(self._import_file)

        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh_page)

        delete_button = QPushButton("删除选中分子")
        delete_button.clicked.connect(self._delete_selected_molecule)

        control_layout.addWidget(QLabel("检索"))
        control_layout.addWidget(self.search_input, stretch=1)
        control_layout.addWidget(import_button)
        control_layout.addWidget(refresh_button)
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
        return panel

    def refresh_page(self) -> None:
        search_text = self.search_input.text().strip() if hasattr(self, "search_input") else ""
        self.dataset = self.db_manager.get_wide_dataset(search_text=search_text)
        self.dataset_model.set_dataframe(self.dataset)
        self.table_view.resizeColumnsToContents()
        self.status_label.setText(
            f"当前记录数: {len(self.dataset)} | 数值列: {len(self.dataset.select_dtypes(include='number').columns)}"
        )
        self._sync_table_selection()

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
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return

        self.refresh_page()
        QMessageBox.information(
            self,
            "导入完成",
            f"已导入 {result['row_count']} 条记录。\n来源文件: {Path(file_path).name}",
        )

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

        self.db_manager.delete_molecule(molecule_id)
        self.refresh_page()

    def _sync_table_selection(self) -> None:
        if self.dataset.empty:
            self.table_view.clearSelection()
            self._render_selected_molecule(error_message="暂无可显示的分子数据。")
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
