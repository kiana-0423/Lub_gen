"""Full periodic table dialog for element selection."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QGridLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from draw.core import PERIODIC_TABLE_GRID, atomic_number_for_symbol


class ElementDialog(QDialog):
    """Dialog that exposes the full periodic table."""

    def __init__(self, current_symbol: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected_symbol = current_symbol
        self.setWindowTitle("More Elements")
        self.resize(960, 420)
        self._build_ui(current_symbol)

    @property
    def selected_symbol(self) -> str:
        """Return the selected element symbol."""
        return self._selected_symbol

    def _build_ui(self, current_symbol: str) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select an element from the periodic table."))

        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)

        row_labels = ("1", "2", "3", "4", "5", "6", "7", "La-Lu", "Ac-Lr")
        for row_index, row_label in enumerate(row_labels):
            grid.addWidget(QLabel(row_label), row_index, 0)

        for row_index, row in enumerate(PERIODIC_TABLE_GRID):
            for column_index, symbol in enumerate(row, start=1):
                if not symbol:
                    continue

                button = QToolButton()
                button.setText(symbol)
                button.setCheckable(True)
                button.setChecked(symbol == current_symbol)
                button.setToolTip(f"{symbol} (Z={atomic_number_for_symbol(symbol)})")
                button.clicked.connect(lambda checked=False, choice=symbol: self._choose(choice))
                grid.addWidget(button, row_index, column_index)

        layout.addLayout(grid)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose(self, symbol: str) -> None:
        self._selected_symbol = symbol
        self.accept()
