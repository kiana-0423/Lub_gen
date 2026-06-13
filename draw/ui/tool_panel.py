"""Left-side editor tool panel."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from draw.core import COMMON_ELEMENT_SYMBOLS
from draw.core.models import BondType
from draw.editor.tools import ALL_TOOLS

from .element_dialog import ElementDialog


class ToolPanel(QWidget):
    """Tool, element, and bond-type controls for the editor shell."""

    tool_selected = Signal(str)
    element_selected = Signal(str)
    bond_type_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_buttons: dict[str, QToolButton] = {}
        self._element_group = QButtonGroup(self)
        self._element_group.setExclusive(True)
        self._element_buttons: dict[str, QToolButton] = {}
        self._bond_group = QButtonGroup(self)
        self._bond_group.setExclusive(True)
        self._bond_buttons: dict[BondType, QToolButton] = {}
        self._current_element_label = QLabel("C")
        self._current_bond_label = QLabel(BondType.SINGLE.display_name)
        self._build_ui()

    def set_active_tool(self, tool_name: str) -> None:
        """Synchronize the selected tool button."""
        button = self._tool_buttons.get(tool_name)
        if button is not None:
            button.setChecked(True)

    def set_current_element(self, symbol: str) -> None:
        """Synchronize the current element selection."""
        self._current_element_label.setText(symbol)
        if symbol in self._element_buttons:
            self._element_buttons[symbol].setChecked(True)
            return
        self._clear_group_selection(self._element_group)

    def set_current_bond_type(self, bond_type: BondType) -> None:
        """Synchronize the current bond-type selection."""
        self._current_bond_label.setText(bond_type.display_name)
        button = self._bond_buttons.get(bond_type)
        if button is not None:
            button.setChecked(True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(self._build_tool_group())
        layout.addWidget(self._build_element_group())
        layout.addWidget(self._build_bond_group())
        layout.addStretch(1)

        hint = QLabel("Use the atom tool to place or retag atoms. Use the bond tool to draw or retag bonds.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def _build_tool_group(self) -> QGroupBox:
        group = QGroupBox("Tools")
        layout = QVBoxLayout(group)

        for tool_name in ALL_TOOLS:
            button = QToolButton()
            button.setText(tool_name)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, name=tool_name: self.tool_selected.emit(name))
            self._tool_group.addButton(button)
            self._tool_buttons[tool_name] = button
            layout.addWidget(button)

        self._tool_buttons["Select"].setChecked(True)
        return group

    def _build_element_group(self) -> QGroupBox:
        group = QGroupBox("Elements")
        layout = QVBoxLayout(group)

        current_row = QHBoxLayout()
        current_row.addWidget(QLabel("Current:"))
        current_row.addWidget(self._current_element_label)
        current_row.addStretch(1)
        layout.addLayout(current_row)

        grid = QGridLayout()
        for index, symbol in enumerate(COMMON_ELEMENT_SYMBOLS):
            button = QToolButton()
            button.setText(symbol)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, value=symbol: self._emit_element(value))
            self._element_group.addButton(button)
            self._element_buttons[symbol] = button
            grid.addWidget(button, index // 2, index % 2)
        layout.addLayout(grid)

        more_button = QPushButton("More Elements...")
        more_button.clicked.connect(self._open_element_dialog)
        layout.addWidget(more_button)

        self._element_buttons["C"].setChecked(True)
        return group

    def _build_bond_group(self) -> QGroupBox:
        group = QGroupBox("Bond Types")
        layout = QVBoxLayout(group)

        current_row = QHBoxLayout()
        current_row.addWidget(QLabel("Current:"))
        current_row.addWidget(self._current_bond_label)
        current_row.addStretch(1)
        layout.addLayout(current_row)

        for bond_type in BondType:
            button = QToolButton()
            button.setText(bond_type.display_name)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, value=bond_type: self._emit_bond_type(value))
            self._bond_group.addButton(button)
            self._bond_buttons[bond_type] = button
            layout.addWidget(button)

        self._bond_buttons[BondType.SINGLE].setChecked(True)
        return group

    def _emit_element(self, symbol: str) -> None:
        self.set_current_element(symbol)
        self.element_selected.emit(symbol)

    def _emit_bond_type(self, bond_type: BondType) -> None:
        self.set_current_bond_type(bond_type)
        self.bond_type_selected.emit(bond_type.value)

    def _open_element_dialog(self) -> None:
        dialog = ElementDialog(self._current_element_label.text(), self)
        if dialog.exec():
            self._emit_element(dialog.selected_symbol)

    @staticmethod
    def _clear_group_selection(group: QButtonGroup) -> None:
        group.setExclusive(False)
        for button in group.buttons():
            button.setChecked(False)
        group.setExclusive(True)
