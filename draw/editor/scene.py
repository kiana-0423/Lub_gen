"""Molecule editor scene and document mutation logic."""

from __future__ import annotations

from copy import deepcopy
from math import hypot

from PySide6.QtCore import QPointF, Signal
from PySide6.QtWidgets import QGraphicsScene

from draw.chemistry_services.base import ChemistryService
from draw.commands.editor_commands import (
    AddAtomCommand,
    AddBondCommand,
    ClearCanvasCommand,
    DeleteSelectionCommand,
    ReplaceDocumentCommand,
    UpdateAtomElementCommand,
    UpdateBondTypeCommand,
)
from draw.commands.stack import EditorCommandStack
from draw.core.elements import element_info_for_atomic_number, element_info_for_symbol
from draw.core.models import Atom, Bond, BondType, MoleculeDocument

from .items import AtomItem, BondItem
from .tools import EditorTool

GRID_SIZE = 25.0
DEFAULT_BOND_LENGTH = 50.0


class MoleculeScene(QGraphicsScene):
    """Scene responsible for editor state, interactions, and chemistry refresh."""

    document_changed = Signal(object)
    selection_summary_changed = Signal(int, int)
    status_message = Signal(str)

    def __init__(self, command_stack: EditorCommandStack, chemistry_service: ChemistryService) -> None:
        super().__init__()
        self._command_stack = command_stack
        self._chemistry_service = chemistry_service
        self._document = MoleculeDocument()
        self._active_tool = EditorTool.SELECT
        self._current_element = element_info_for_symbol("C")
        self._current_bond_type = BondType.SINGLE
        self._pending_bond_atom_id: int | None = None
        self._atom_items: dict[int, AtomItem] = {}
        self._bond_items: dict[int, BondItem] = {}
        self._rebuilding = False
        self.selectionChanged.connect(self._on_selection_changed)

    @property
    def active_tool(self) -> str:
        """Return the active editor tool label."""
        return self._active_tool.value

    @property
    def current_element_symbol(self) -> str:
        """Return the current element symbol used by the atom tool."""
        return self._current_element.symbol

    @property
    def current_bond_type(self) -> BondType:
        """Return the current bond type used by the bond tool."""
        return self._current_bond_type

    @property
    def invalid_atom_count(self) -> int:
        """Return the number of atoms currently flagged as invalid."""
        return sum(1 for atom in self._document.atoms.values() if not atom.is_valid)

    def set_active_tool(self, tool_name: str) -> None:
        """Switch the active editor tool."""
        self._active_tool = EditorTool(tool_name)
        if self._active_tool is not EditorTool.BOND:
            self._pending_bond_atom_id = None

    def set_current_element(self, symbol: str) -> None:
        """Set the element used by the atom tool."""
        self._current_element = element_info_for_symbol(symbol)

    def set_current_bond_type(self, bond_type: BondType | str) -> None:
        """Set the bond type used by the bond tool."""
        self._current_bond_type = bond_type if isinstance(bond_type, BondType) else BondType(bond_type)

    def document_snapshot(self) -> MoleculeDocument:
        """Return a detached snapshot of the document."""
        return self._document.clone()

    def is_empty(self) -> bool:
        """Return whether the scene currently has any chemistry items."""
        return self._document.atom_count == 0 and self._document.bond_count == 0

    def post_status(self, message: str) -> None:
        """Emit an editor status message."""
        self.status_message.emit(message)

    def handle_primary_click(self, scene_pos: QPointF) -> bool:
        """Process a left-click according to the active tool."""
        if self._active_tool is EditorTool.SELECT:
            return False

        if self._active_tool is EditorTool.ATOM:
            atom_item = self._atom_item_at(scene_pos)
            if atom_item is not None:
                atom = self._document.get_atom(atom_item.atom_id)
                if atom is not None and atom.atomic_number == self._current_element.atomic_number:
                    self.post_status(f"Atom {atom.atom_id} is already {atom.symbol}.")
                    return True
                self._command_stack.push(
                    UpdateAtomElementCommand(self, atom_item.atom_id, self._current_element.atomic_number)
                )
                return True

            snapped = self.snap_to_grid(scene_pos)
            self._command_stack.push(AddAtomCommand(self, snapped, atomic_number=self._current_element.atomic_number))
            return True

        if self._active_tool is EditorTool.BOND:
            bond_item = self._bond_item_at(scene_pos)
            if bond_item is not None:
                bond = self._document.get_bond(bond_item.bond_id)
                if bond is not None and bond.bond_type is self._current_bond_type:
                    self.post_status(f"Bond {bond.bond_id} is already {bond.bond_type.display_name.lower()}.")
                    return True
                self._pending_bond_atom_id = None
                self._command_stack.push(UpdateBondTypeCommand(self, bond_item.bond_id, self._current_bond_type))
                return True

            self._handle_bond_click(scene_pos)
            return True

        if self._active_tool is EditorTool.DELETE:
            target_item = self._pick_deletable_item(scene_pos)
            if target_item is None:
                self.post_status("Nothing to delete at this position.")
                return True
            self.clearSelection()
            target_item.setSelected(True)
            self.delete_selected()
            return True

        return False

    def delete_selected(self) -> None:
        """Delete the currently selected atoms and bonds."""
        atom_ids, bond_ids = self.selected_ids()
        if not atom_ids and not bond_ids:
            self.post_status("Nothing selected to delete.")
            return

        fragment = self.capture_fragment(atom_ids, bond_ids)
        self._command_stack.push(DeleteSelectionCommand(self, fragment))

    def clear_document(self) -> None:
        """Clear the full canvas with undo support."""
        if self.is_empty():
            self.post_status("Canvas is already empty.")
            return
        self._command_stack.push(ClearCanvasCommand(self, self.document_snapshot()))

    def apply_document_change(self, document: MoleculeDocument, command_text: str) -> None:
        """Apply a whole-document transformation as an undoable command."""
        self._command_stack.push(ReplaceDocumentCommand(self, document, command_text))

    def add_atom(
        self,
        position: QPointF,
        atomic_number: int,
        *,
        preferred_atom_id: int | None = None,
        select_created: bool = False,
    ) -> Atom:
        """Create an atom in the document."""
        element = element_info_for_atomic_number(atomic_number)
        snapped = self.snap_to_grid(position)
        atom = Atom(
            atom_id=self._document.allocate_atom_id(preferred_atom_id),
            atomic_number=element.atomic_number,
            symbol=element.symbol,
            x=snapped.x(),
            y=snapped.y(),
        )
        self._document.add_atom(atom)
        self._after_document_mutation(select_atom_ids=[atom.atom_id] if select_created else None)
        return self._document.atoms[atom.atom_id]

    def replace_atom(self, atom: Atom, *, select_updated: bool = False) -> Atom:
        """Replace an atom in the document."""
        self._document.add_atom(atom.normalized())
        self._after_document_mutation(select_atom_ids=[atom.atom_id] if select_updated else None)
        return self._document.atoms[atom.atom_id]

    def update_atom_element(self, atom_id: int, atomic_number: int, *, select_updated: bool = False) -> Atom:
        """Change the element of an existing atom."""
        atom = self._document.get_atom(atom_id)
        if atom is None:
            raise RuntimeError("Cannot update an atom that does not exist.")

        element = element_info_for_atomic_number(atomic_number)
        atom.atomic_number = element.atomic_number
        atom.symbol = element.symbol
        atom.aromatic = atom.atom_id in self._aromatic_atom_ids_from_bonds()
        self._after_document_mutation(select_atom_ids=[atom_id] if select_updated else None)
        return self._document.atoms[atom_id]

    def add_bond(
        self,
        atom_a_id: int,
        atom_b_id: int,
        bond_type: BondType,
        *,
        preferred_bond_id: int | None = None,
        select_created: bool = False,
    ) -> Bond:
        """Create a bond between two existing atoms."""
        atom_a = self._document.get_atom(atom_a_id)
        atom_b = self._document.get_atom(atom_b_id)
        if atom_a is None or atom_b is None:
            raise RuntimeError("Cannot create a bond without two atoms.")
        if self._document.has_bond_between(atom_a_id, atom_b_id):
            raise RuntimeError("A bond between these atoms already exists.")

        bond = Bond(
            bond_id=self._document.allocate_bond_id(preferred_bond_id),
            atom_a_id=atom_a_id,
            atom_b_id=atom_b_id,
            bond_type=bond_type,
        )
        self._document.add_bond(bond)
        self._after_document_mutation(select_bond_ids=[bond.bond_id] if select_created else None)
        return self._document.bonds[bond.bond_id]

    def add_bond_with_new_atom(
        self,
        *,
        atom_a_id: int,
        atom_b_position: QPointF,
        atom_b_atomic_number: int,
        bond_type: BondType,
        preferred_atom_id: int | None = None,
        preferred_bond_id: int | None = None,
        select_created: bool = False,
    ) -> tuple[Atom, Bond]:
        """Create a new atom and immediately bond it to an existing atom."""
        element = element_info_for_atomic_number(atom_b_atomic_number)
        snapped = self.snap_to_grid(atom_b_position)
        atom = Atom(
            atom_id=self._document.allocate_atom_id(preferred_atom_id),
            atomic_number=element.atomic_number,
            symbol=element.symbol,
            x=snapped.x(),
            y=snapped.y(),
        )
        bond = Bond(
            bond_id=self._document.allocate_bond_id(preferred_bond_id),
            atom_a_id=atom_a_id,
            atom_b_id=atom.atom_id,
            bond_type=bond_type,
        )
        self._document.add_atom(atom)
        self._document.add_bond(bond)
        self._after_document_mutation(
            select_atom_ids=[atom.atom_id] if select_created else None,
            select_bond_ids=[bond.bond_id] if select_created else None,
        )
        return self._document.atoms[atom.atom_id], self._document.bonds[bond.bond_id]

    def replace_bond(self, bond: Bond, *, select_updated: bool = False) -> Bond:
        """Replace a bond in the document."""
        self._document.add_bond(deepcopy(bond))
        self._after_document_mutation(select_bond_ids=[bond.bond_id] if select_updated else None)
        return self._document.bonds[bond.bond_id]

    def update_bond_type(self, bond_id: int, bond_type: BondType, *, select_updated: bool = False) -> Bond:
        """Change the type of an existing bond."""
        bond = self._document.get_bond(bond_id)
        if bond is None:
            raise RuntimeError("Cannot update a bond that does not exist.")
        bond.bond_type = bond_type
        self._after_document_mutation(select_bond_ids=[bond_id] if select_updated else None)
        return self._document.bonds[bond_id]

    def remove_atom(self, atom_id: int) -> Atom | None:
        """Remove an atom and any bonds attached to it."""
        for bond_id in list(self._document.bond_ids_for_atom(atom_id)):
            self._document.remove_bond(bond_id)
        atom = self._document.remove_atom(atom_id)
        if self._pending_bond_atom_id == atom_id:
            self._pending_bond_atom_id = None
        self._after_document_mutation()
        return atom

    def remove_bond(self, bond_id: int) -> Bond | None:
        """Remove a bond from the document."""
        bond = self._document.remove_bond(bond_id)
        self._after_document_mutation()
        return bond

    def load_document(self, document: MoleculeDocument) -> None:
        """Replace the current document and rebuild the scene."""
        self._pending_bond_atom_id = None
        self._document = document.clone()
        self._after_document_mutation()

    def capture_fragment(self, atom_ids: list[int], bond_ids: list[int]) -> MoleculeDocument:
        """Capture a restorable fragment for deletion commands."""
        fragment = MoleculeDocument(
            next_atom_id=self._document.next_atom_id,
            next_bond_id=self._document.next_bond_id,
        )

        expanded_atom_ids = {atom_id for atom_id in atom_ids if self._document.get_atom(atom_id) is not None}
        expanded_bond_ids = {bond_id for bond_id in bond_ids if self._document.get_bond(bond_id) is not None}

        for atom_id in expanded_atom_ids:
            expanded_bond_ids.update(self._document.bond_ids_for_atom(atom_id))

        for atom_id in sorted(expanded_atom_ids):
            atom = self._document.get_atom(atom_id)
            if atom is not None:
                fragment.add_atom(deepcopy(atom))

        for bond_id in sorted(expanded_bond_ids):
            bond = self._document.get_bond(bond_id)
            if bond is not None:
                fragment.add_bond(deepcopy(bond))

        return fragment

    def delete_fragment(self, fragment: MoleculeDocument) -> None:
        """Delete a previously captured fragment from the scene."""
        if self._pending_bond_atom_id in fragment.atoms:
            self._pending_bond_atom_id = None

        for bond_id in list(fragment.bonds):
            self._document.remove_bond(bond_id)
        for atom_id in list(fragment.atoms):
            self._document.remove_atom(atom_id)

        self._after_document_mutation()

    def restore_fragment(self, fragment: MoleculeDocument) -> None:
        """Restore a previously deleted fragment."""
        for atom in sorted(fragment.atoms.values(), key=lambda item: item.atom_id):
            self._document.add_atom(deepcopy(atom).normalized())

        for bond in sorted(fragment.bonds.values(), key=lambda item: item.bond_id):
            self._document.add_bond(deepcopy(bond))

        self._after_document_mutation()

    def selected_ids(self) -> tuple[list[int], list[int]]:
        """Return the currently selected atom and bond identifiers."""
        atom_ids: list[int] = []
        bond_ids: list[int] = []

        for item in self.selectedItems():
            if isinstance(item, AtomItem):
                atom_ids.append(item.atom_id)
            elif isinstance(item, BondItem):
                bond_ids.append(item.bond_id)

        return atom_ids, bond_ids

    def snap_to_grid(self, point: QPointF) -> QPointF:
        """Snap a point to the editor grid."""
        return QPointF(
            round(point.x() / GRID_SIZE) * GRID_SIZE,
            round(point.y() / GRID_SIZE) * GRID_SIZE,
        )

    def _handle_bond_click(self, scene_pos: QPointF) -> None:
        atom_item = self._atom_item_at(scene_pos)
        if self._pending_bond_atom_id is None:
            if atom_item is None:
                self.post_status("Click an existing atom to start a bond.")
                return

            self._pending_bond_atom_id = atom_item.atom_id
            self.clearSelection()
            atom_item.setSelected(True)
            self.post_status(f"Bond start set to atom {atom_item.atom_id}.")
            return

        start_atom_id = self._pending_bond_atom_id
        if atom_item is not None:
            if atom_item.atom_id == start_atom_id:
                self.post_status("Choose a different atom for the bond target.")
                return

            existing_bond = self._document.find_bond_between(start_atom_id, atom_item.atom_id)
            if existing_bond is not None:
                if existing_bond.bond_type is self._current_bond_type:
                    self.post_status(
                        f"Bond {existing_bond.bond_id} is already {existing_bond.bond_type.display_name.lower()}."
                    )
                else:
                    self._command_stack.push(
                        UpdateBondTypeCommand(self, existing_bond.bond_id, self._current_bond_type)
                    )
                self._pending_bond_atom_id = None
                return

            self._command_stack.push(
                AddBondCommand(
                    self,
                    atom_a_id=start_atom_id,
                    atom_b_id=atom_item.atom_id,
                    bond_type=self._current_bond_type,
                )
            )
            self._pending_bond_atom_id = None
            return

        target_position = self._resolve_bond_target_position(start_atom_id, scene_pos)
        self._command_stack.push(
            AddBondCommand(
                self,
                atom_a_id=start_atom_id,
                atom_b_id=None,
                atom_b_position=target_position,
                atom_b_atomic_number=self._current_element.atomic_number,
                bond_type=self._current_bond_type,
            )
        )
        self._pending_bond_atom_id = None

    def _resolve_bond_target_position(self, atom_id: int, scene_pos: QPointF) -> QPointF:
        """Resolve the location for a newly created atom during bond placement."""
        snapped = self.snap_to_grid(scene_pos)
        anchor = self._document.get_atom(atom_id)
        if anchor is None:
            return snapped

        if hypot(anchor.x - snapped.x(), anchor.y - snapped.y()) < GRID_SIZE / 2:
            return QPointF(anchor.x + DEFAULT_BOND_LENGTH, anchor.y)
        return snapped

    def _atom_item_at(self, scene_pos: QPointF) -> AtomItem | None:
        for item in self.items(scene_pos):
            if isinstance(item, AtomItem):
                return item
        return None

    def _bond_item_at(self, scene_pos: QPointF) -> BondItem | None:
        for item in self.items(scene_pos):
            if isinstance(item, BondItem):
                return item
        return None

    def _pick_deletable_item(self, scene_pos: QPointF) -> AtomItem | BondItem | None:
        for item in self.items(scene_pos):
            if isinstance(item, (AtomItem, BondItem)):
                return item
        return None

    def _after_document_mutation(
        self,
        *,
        select_atom_ids: list[int] | None = None,
        select_bond_ids: list[int] | None = None,
    ) -> None:
        self._normalize_document_metadata()
        self._refresh_document_state()
        self._rebuild_scene_items(
            selected_atom_ids=set(select_atom_ids or []),
            selected_bond_ids=set(select_bond_ids or []),
        )
        self._emit_document_changed()

    def _normalize_document_metadata(self) -> None:
        aromatic_atom_ids = self._aromatic_atom_ids_from_bonds()
        for atom_id, atom in list(self._document.atoms.items()):
            normalized = atom.normalized()
            normalized.aromatic = atom_id in aromatic_atom_ids
            self._document.atoms[atom_id] = normalized

    def _refresh_document_state(self) -> None:
        try:
            self._document = self._chemistry_service.refresh_document_state(self._document)
        except Exception as exc:
            self.post_status(f"Chemistry refresh warning: {exc}")

    def _rebuild_scene_items(self, *, selected_atom_ids: set[int], selected_bond_ids: set[int]) -> None:
        self._rebuilding = True
        self.clearSelection()
        super().clear()
        self._atom_items.clear()
        self._bond_items.clear()

        for atom in sorted(self._document.atoms.values(), key=lambda item: item.atom_id):
            atom_item = AtomItem(atom)
            self.addItem(atom_item)
            self._atom_items[atom.atom_id] = atom_item

        for bond in sorted(self._document.bonds.values(), key=lambda item: item.bond_id):
            atom_a = self._document.get_atom(bond.atom_a_id)
            atom_b = self._document.get_atom(bond.atom_b_id)
            if atom_a is None or atom_b is None:
                continue

            bond_item = BondItem(
                bond=bond,
                start_point=QPointF(atom_a.x, atom_a.y),
                end_point=QPointF(atom_b.x, atom_b.y),
            )
            self.addItem(bond_item)
            self._bond_items[bond.bond_id] = bond_item

        for atom_id in selected_atom_ids:
            atom_item = self._atom_items.get(atom_id)
            if atom_item is not None:
                atom_item.setSelected(True)

        for bond_id in selected_bond_ids:
            bond_item = self._bond_items.get(bond_id)
            if bond_item is not None:
                bond_item.setSelected(True)

        self._rebuilding = False
        self._refresh_item_styles()

    def _emit_document_changed(self) -> None:
        self.document_changed.emit(self._document.clone())
        atom_ids, bond_ids = self.selected_ids()
        self.selection_summary_changed.emit(len(atom_ids), len(bond_ids))

    def _on_selection_changed(self) -> None:
        if self._rebuilding:
            return
        self._refresh_item_styles()
        atom_ids, bond_ids = self.selected_ids()
        self.selection_summary_changed.emit(len(atom_ids), len(bond_ids))

    def _refresh_item_styles(self) -> None:
        for atom_item in self._atom_items.values():
            atom_item.update_style()
        for bond_item in self._bond_items.values():
            bond_item.update_style()

    def _aromatic_atom_ids_from_bonds(self) -> set[int]:
        return {
            atom_id
            for bond in self._document.bonds.values()
            if bond.bond_type is BondType.AROMATIC
            for atom_id in (bond.atom_a_id, bond.atom_b_id)
        }
