from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
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

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.services.data_import_service import DataImportService
from chemstudio.services.visualization_service import VisualizationService
from chemstudio.ui.widgets import BasePage, MatplotlibCanvas, PandasTableModel


class DataPage(BasePage):
    """Data import, browsing, filtering, and visualization page."""

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
            "导入 CSV/Excel，保存至 SQLite，浏览数据，并完成基础分布图、散点图和缺失值统计。",
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
        splitter.addWidget(self._build_plot_panel())
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

        layout.addWidget(self.table_view)
        return panel

    def _build_plot_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        distribution_box = QGroupBox("数值列分布图")
        distribution_layout = QHBoxLayout(distribution_box)
        self.distribution_column_input = QLineEdit()
        self.distribution_column_input.setPlaceholderText("输入数值列名")
        distribution_button = QPushButton("绘制分布")
        distribution_button.clicked.connect(self._plot_distribution)
        distribution_layout.addWidget(self.distribution_column_input, stretch=1)
        distribution_layout.addWidget(distribution_button)

        scatter_box = QGroupBox("两列散点图")
        scatter_layout = QGridLayout(scatter_box)
        self.scatter_x_input = QLineEdit()
        self.scatter_x_input.setPlaceholderText("X 列名")
        self.scatter_y_input = QLineEdit()
        self.scatter_y_input.setPlaceholderText("Y 列名")
        scatter_button = QPushButton("绘制散点")
        scatter_button.clicked.connect(self._plot_scatter)
        missing_button = QPushButton("缺失值统计")
        missing_button.clicked.connect(self._plot_missing_values)
        scatter_layout.addWidget(QLabel("X"), 0, 0)
        scatter_layout.addWidget(self.scatter_x_input, 0, 1)
        scatter_layout.addWidget(QLabel("Y"), 1, 0)
        scatter_layout.addWidget(self.scatter_y_input, 1, 1)
        scatter_layout.addWidget(scatter_button, 2, 0)
        scatter_layout.addWidget(missing_button, 2, 1)

        self.canvas = MatplotlibCanvas(width=5.2, height=4.2)

        layout.addWidget(distribution_box)
        layout.addWidget(scatter_box)
        layout.addWidget(self.canvas, stretch=1)
        return panel

    def refresh_page(self) -> None:
        search_text = self.search_input.text().strip() if hasattr(self, "search_input") else ""
        self.dataset = self.db_manager.get_wide_dataset(search_text=search_text)
        self.dataset_model.set_dataframe(self.dataset)
        self.table_view.resizeColumnsToContents()
        self.status_label.setText(
            f"当前记录数: {len(self.dataset)} | 数值列: {len(self.dataset.select_dtypes(include='number').columns)}"
        )

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

    def _plot_distribution(self) -> None:
        column_name = self.distribution_column_input.text().strip()
        if column_name not in self.dataset.columns:
            QMessageBox.warning(self, "列不存在", "请输入有效的数值列名。")
            return
        self.visualization_service.plot_distribution(self.canvas.axes, self.dataset, column_name)
        self.canvas.draw_idle()

    def _plot_scatter(self) -> None:
        x_column = self.scatter_x_input.text().strip()
        y_column = self.scatter_y_input.text().strip()
        if x_column not in self.dataset.columns or y_column not in self.dataset.columns:
            QMessageBox.warning(self, "列不存在", "请输入有效的 X/Y 列名。")
            return
        self.visualization_service.plot_scatter(self.canvas.axes, self.dataset, x_column, y_column)
        self.canvas.draw_idle()

    def _plot_missing_values(self) -> None:
        self.visualization_service.plot_missing_values(self.canvas.axes, self.dataset)
        self.canvas.draw_idle()
