from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
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

        self._summary_pixmap: QPixmap | None = None

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.addStretch()
        self.open_summary_button = QPushButton("放大查看")
        self.open_summary_button.setEnabled(False)
        self.open_summary_button.clicked.connect(self._open_summary_dialog)
        toolbar_layout.addWidget(self.open_summary_button)
        layout.addLayout(toolbar_layout)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.summary_label = QLabel("尚未生成模型解释。")
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_label.setMinimumHeight(220)
        self.summary_label.setScaledContents(False)
        self.summary_label.setToolTip("双击 SHAP 图可放大查看")
        self.summary_label.installEventFilter(self)

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
        self._summary_pixmap = None
        self.open_summary_button.setEnabled(False)
        if image_path.is_file():
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                self._summary_pixmap = pixmap
                self.open_summary_button.setEnabled(True)
                self._refresh_summary_preview()
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

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched is self.summary_label and event.type() == QEvent.Type.MouseButtonDblClick:
            self._open_summary_dialog()
            return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_summary_preview()

    def _refresh_summary_preview(self) -> None:
        if self._summary_pixmap is None or self._summary_pixmap.isNull():
            return
        self.summary_label.setPixmap(
            self._summary_pixmap.scaled(
                self.summary_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _open_summary_dialog(self) -> None:
        if self._summary_pixmap is None or self._summary_pixmap.isNull():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("SHAP 图放大查看")
        dialog.resize(1100, 760)

        layout = QVBoxLayout(dialog)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setPixmap(self._summary_pixmap)
        image_label.resize(self._summary_pixmap.size())

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(image_label)
        layout.addWidget(scroll_area)

        dialog.exec()


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
