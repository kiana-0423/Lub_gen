"""Graphics items used by the molecule editor scene."""

from __future__ import annotations

from math import hypot

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsSimpleTextItem,
)

from draw.core.models import Atom, Bond, BondType

ATOM_RADIUS = 18.0
BOND_OFFSET = 5.0


class AtomItem(QGraphicsEllipseItem):
    """Selectable atom node item."""

    def __init__(self, atom: Atom) -> None:
        super().__init__(-ATOM_RADIUS, -ATOM_RADIUS, ATOM_RADIUS * 2, ATOM_RADIUS * 2)
        self.atom_id = atom.atom_id
        self._atom = atom
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setZValue(2)

        self._label = QGraphicsSimpleTextItem("", self)
        self._hydrogen_label = QGraphicsSimpleTextItem("", self)
        self.update_from_atom(atom)

    def update_from_atom(self, atom: Atom) -> None:
        """Refresh the displayed atom data."""
        self.atom_id = atom.atom_id
        self._atom = atom
        self.setPos(atom.x, atom.y)
        self._label.setText(self._label_text(atom))
        self._hydrogen_label.setText("")
        self._center_labels()
        self.setToolTip(self._tooltip_text(atom))
        self.update_style()

    def update_style(self) -> None:
        """Refresh the atom appearance based on chemistry state and selection state."""
        atom = self._atom
        if atom.is_valid:
            if atom.aromatic:
                pen_color = QColor("#a16207")
                brush_color = QColor("#fef3c7")
            else:
                pen_color = QColor("#0f172a")
                brush_color = QColor("#ffffff")
        else:
            pen_color = QColor("#b91c1c")
            brush_color = QColor("#fee2e2")

        if self.isSelected():
            pen_color = QColor("#0f766e") if atom.is_valid else QColor("#7f1d1d")
            brush_color = QColor("#ccfbf1") if atom.is_valid else QColor("#fecaca")

        self.setPen(QPen(pen_color, 2.5))
        self.setBrush(brush_color)
        self._label.setBrush(QColor("#0f172a") if atom.is_valid else QColor("#7f1d1d"))
        self._hydrogen_label.setBrush(QColor("#475569") if atom.is_valid else QColor("#991b1b"))

    def position(self) -> QPointF:
        """Return the scene position for the atom."""
        return self.scenePos()

    def _center_labels(self) -> None:
        label_rect = self._label.boundingRect()
        self._label.setPos(-label_rect.width() / 2, -label_rect.height() / 2 - 4)
        self._hydrogen_label.setVisible(False)

    @staticmethod
    def _label_text(atom: Atom) -> str:
        return atom.symbol

    @staticmethod
    def _tooltip_text(atom: Atom) -> str:
        parts = [
            f"{atom.symbol} (Z={atom.atomic_number})",
            f"Formal charge: {atom.formal_charge}",
            f"Implicit H: {atom.implicit_hydrogens}",
            f"Aromatic: {'yes' if atom.aromatic else 'no'}",
            f"Valid: {'yes' if atom.is_valid else 'no'}",
        ]
        if atom.validation_warning:
            parts.append(f"Warning: {atom.validation_warning}")
        return "\n".join(parts)


class BondItem(QGraphicsPathItem):
    """Selectable bond edge item with bond-type-aware rendering."""

    def __init__(self, bond: Bond, start_point: QPointF, end_point: QPointF) -> None:
        super().__init__()
        self.bond_id = bond.bond_id
        self.atom_a_id = bond.atom_a_id
        self.atom_b_id = bond.atom_b_id
        self._bond = bond
        self._start_point = start_point
        self._end_point = end_point
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setZValue(1)
        self.update_from_bond(bond, start_point, end_point)

    def update_from_bond(self, bond: Bond, start_point: QPointF, end_point: QPointF) -> None:
        """Refresh the displayed bond data."""
        self.bond_id = bond.bond_id
        self.atom_a_id = bond.atom_a_id
        self.atom_b_id = bond.atom_b_id
        self._bond = bond
        self._start_point = start_point
        self._end_point = end_point
        self.setPath(self._build_path(start_point, end_point, bond.bond_type))
        self.setToolTip(f"{bond.bond_type.display_name} bond")
        self.update_style()

    def update_style(self) -> None:
        """Refresh the bond appearance based on bond type and selection state."""
        color = QColor("#92400e") if self._bond.bond_type is BondType.AROMATIC else QColor("#334155")
        width = 3.5 if self._bond.bond_type in {BondType.SINGLE, BondType.AROMATIC} else 2.3

        if self.isSelected():
            color = QColor("#0f766e")
            width = max(width, 3.5)

        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if self._bond.bond_type is BondType.AROMATIC:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)

    def shape(self) -> QPainterPath:
        """Widen the clickable area so bonds remain easy to select."""
        stroker = QPainterPathStroker()
        stroker.setWidth(14)
        return stroker.createStroke(self.path())

    @staticmethod
    def _build_path(start_point: QPointF, end_point: QPointF, bond_type: BondType) -> QPainterPath:
        path = QPainterPath()
        dx = end_point.x() - start_point.x()
        dy = end_point.y() - start_point.y()
        length = hypot(dx, dy)

        if length == 0:
            path.moveTo(start_point)
            path.lineTo(end_point)
            return path

        perp_x = -dy / length
        perp_y = dx / length

        def offset_point(point: QPointF, offset: float) -> QPointF:
            return QPointF(point.x() + perp_x * offset, point.y() + perp_y * offset)

        if bond_type is BondType.DOUBLE:
            BondItem._append_line(path, offset_point(start_point, BOND_OFFSET), offset_point(end_point, BOND_OFFSET))
            BondItem._append_line(path, offset_point(start_point, -BOND_OFFSET), offset_point(end_point, -BOND_OFFSET))
            return path

        if bond_type is BondType.TRIPLE:
            BondItem._append_line(path, start_point, end_point)
            BondItem._append_line(path, offset_point(start_point, BOND_OFFSET * 1.5), offset_point(end_point, BOND_OFFSET * 1.5))
            BondItem._append_line(path, offset_point(start_point, -BOND_OFFSET * 1.5), offset_point(end_point, -BOND_OFFSET * 1.5))
            return path

        BondItem._append_line(path, start_point, end_point)
        return path

    @staticmethod
    def _append_line(path: QPainterPath, start_point: QPointF, end_point: QPointF) -> None:
        path.moveTo(start_point)
        path.lineTo(end_point)
