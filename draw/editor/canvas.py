"""Central editor canvas view."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QGraphicsView

from draw.chemistry_services.base import ChemistryService
from draw.commands.stack import EditorCommandStack
from draw.core.models import BondType, MoleculeDocument

from .scene import MoleculeScene
from .tools import EditorTool


class EditorCanvas(QGraphicsView):
    """Graphics view for the molecule editor MVP."""

    status_message = Signal(str)
    document_changed = Signal(object)
    selection_summary_changed = Signal(int, int)

    def __init__(self, command_stack: EditorCommandStack, chemistry_service: ChemistryService) -> None:
        self._scene = MoleculeScene(command_stack, chemistry_service)
        super().__init__(self._scene)
        self._configure_view()
        self._connect_scene()

    @property
    def active_tool(self) -> str:
        """Return the current canvas tool name."""
        return self._scene.active_tool

    @property
    def current_element_symbol(self) -> str:
        """Return the current atom element symbol."""
        return self._scene.current_element_symbol

    @property
    def current_bond_type_label(self) -> str:
        """Return the current bond type label."""
        return self._scene.current_bond_type.display_name

    @property
    def current_bond_type(self) -> BondType:
        """Return the current bond type."""
        return self._scene.current_bond_type

    def set_active_tool(self, tool_name: str) -> None:
        """Switch the active tool and update view behavior."""
        self._scene.set_active_tool(tool_name)
        if tool_name == EditorTool.SELECT.value:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.status_message.emit(f"Active tool changed to {tool_name}.")

    def clear_canvas(self) -> None:
        """Clear the scene through the editor command layer."""
        self._scene.clear_document()

    def delete_selected(self) -> None:
        """Delete the current selection through the editor command layer."""
        self._scene.delete_selected()

    def set_current_element(self, symbol: str) -> None:
        """Switch the current atom element."""
        self._scene.set_current_element(symbol)
        self.viewport().update()
        self.status_message.emit(f"Current element set to {symbol}.")

    def set_current_bond_type(self, bond_type: BondType | str) -> None:
        """Switch the current bond type."""
        self._scene.set_current_bond_type(bond_type)
        self.viewport().update()
        self.status_message.emit(f"Current bond type set to {self._scene.current_bond_type.display_name.lower()}.")

    def zoom_in(self) -> None:
        """Zoom into the scene."""
        self.scale(1.2, 1.2)
        self.status_message.emit(f"Zoomed in to {self._zoom_percent()}%.")

    def zoom_out(self) -> None:
        """Zoom out of the scene."""
        self.scale(1 / 1.2, 1 / 1.2)
        self.status_message.emit(f"Zoomed out to {self._zoom_percent()}%.")

    def document_snapshot(self) -> MoleculeDocument:
        """Return the current document snapshot."""
        return self._scene.document_snapshot()

    def load_document(self, document: MoleculeDocument, *, fit_view: bool = True) -> None:
        """Replace the current scene document."""
        self._scene.load_document(document)
        if fit_view:
            self.resetTransform()
            self._fit_to_content()

    def apply_document_change(self, document: MoleculeDocument, command_text: str, *, fit_view: bool = True) -> None:
        """Apply a whole-document change through the undo stack."""
        self._scene.apply_document_change(document, command_text)
        if fit_view:
            self._fit_to_content()

    def _configure_view(self) -> None:
        self.setSceneRect(-520, -360, 1040, 720)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setBackgroundBrush(QColor("#f7f8fb"))

    def _connect_scene(self) -> None:
        self._scene.status_message.connect(self.status_message.emit)
        self._scene.document_changed.connect(self.document_changed.emit)
        self._scene.selection_summary_changed.connect(self.selection_summary_changed.emit)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Intercept left clicks for editor tool interactions."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            if self._scene.handle_primary_click(scene_pos):
                event.accept()
                return
        super().mousePressEvent(event)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        super().drawBackground(painter, rect)

        grid_pen = QPen(QColor("#e2e8f0"))
        grid_pen.setWidth(1)
        painter.setPen(grid_pen)

        left = int(rect.left()) - (int(rect.left()) % 25)
        top = int(rect.top()) - (int(rect.top()) % 25)

        for x_pos in range(left, int(rect.right()), 25):
            painter.drawLine(x_pos, int(rect.top()), x_pos, int(rect.bottom()))
        for y_pos in range(top, int(rect.bottom()), 25):
            painter.drawLine(int(rect.left()), y_pos, int(rect.right()), y_pos)

        axis_pen = QPen(QColor("#cbd5e1"))
        axis_pen.setWidth(2)
        painter.setPen(axis_pen)
        painter.drawLine(0, int(rect.top()), 0, int(rect.bottom()))
        painter.drawLine(int(rect.left()), 0, int(rect.right()), 0)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # type: ignore[override]
        super().drawForeground(painter, rect)

        info_rect = QRectF(rect.left() + 16, rect.top() + 16, 360, 110)
        painter.setPen(QColor("#0f172a"))
        painter.drawText(
            info_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            (
                f"Tool: {self.active_tool}\n"
                f"Element: {self.current_element_symbol}\n"
                f"Bond: {self.current_bond_type_label}\n"
                f"Atoms: {self._scene.document_snapshot().atom_count}\n"
                f"Bonds: {self._scene.document_snapshot().bond_count}\n"
                f"Invalid atoms: {self._scene.invalid_atom_count}"
            ),
        )

        if self._scene.is_empty():
            empty_rect = QRectF(-240, -36, 480, 72)
            painter.setPen(QColor("#64748b"))
            painter.drawText(
                empty_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Use the Atom tool to place atoms.\nUse the Bond tool to connect them.",
            )

    def _zoom_percent(self) -> int:
        return int(self.transform().m11() * 100)

    def _fit_to_content(self) -> None:
        items_rect = self._scene.itemsBoundingRect()
        if items_rect.isNull():
            return
        self.fitInView(items_rect.adjusted(-80, -80, 80, 80), Qt.AspectRatioMode.KeepAspectRatio)
