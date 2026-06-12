from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from chemstudio.ml.explainer import SHAPExplanation


class SHAPSummaryWidget(QWidget):
    """Widget showing a SHAP summary plot and global feature-importance table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.summary_label = QLabel("尚未生成模型解释。")
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_label.setMinimumHeight(220)
        self.summary_label.setScaledContents(False)

        self.importance_table = QTableWidget(0, 3)
        self.importance_table.setHorizontalHeaderLabels(["排名", "特征名", "SHAP 重要性"])
        self.importance_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.importance_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.importance_table.verticalHeader().setVisible(False)
        self.importance_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.importance_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.importance_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        splitter.addWidget(self.summary_label)
        splitter.addWidget(self.importance_table)
        splitter.setSizes([280, 180])
        layout.addWidget(splitter)

    def load_explanation(self, explanation: SHAPExplanation) -> None:
        """Load a SHAP explanation into the plot label and ranking table."""
        image_path = Path(explanation.summary_plot_path)
        if image_path.is_file():
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                self.summary_label.setPixmap(
                    pixmap.scaled(
                        self.summary_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        else:
            self.summary_label.setText("摘要图文件不存在。")

        rows = list(explanation.global_importance.items())
        self.importance_table.setRowCount(len(rows))
        for row_index, (feature_name, importance) in enumerate(rows):
            rank_item = QTableWidgetItem(str(row_index + 1))
            rank_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_item = QTableWidgetItem(f"{float(importance):.6g}")
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.importance_table.setItem(row_index, 0, rank_item)
            self.importance_table.setItem(row_index, 1, QTableWidgetItem(str(feature_name)))
            self.importance_table.setItem(row_index, 2, value_item)


class SHAPForceWidget(QWidget):
    """Widget showing local SHAP contribution HTML for one prediction."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        layout.addWidget(self.browser)

    def load_force_plot(self, html_content: str) -> None:
        """Display compact force-plot-like HTML content."""
        self.browser.setHtml(html_content)
