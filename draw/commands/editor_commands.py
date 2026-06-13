"""Undoable editor commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF
from PySide6.QtGui import QUndoCommand

from draw.core.models import BondType, MoleculeDocument

if TYPE_CHECKING:
    from draw.editor.scene import MoleculeScene


class AddAtomCommand(QUndoCommand):
    """Create an atom at a fixed position."""

    def __init__(self, scene: "MoleculeScene", position: QPointF, atomic_number: int) -> None:
        super().__init__("Add Atom")
        self._scene = scene
        self._position = QPointF(position)
        self._atomic_number = atomic_number
        self._atom_id: int | None = None

    def redo(self) -> None:
        atom = self._scene.add_atom(
            position=self._position,
            atomic_number=self._atomic_number,
            preferred_atom_id=self._atom_id,
            select_created=True,
        )
        self._atom_id = atom.atom_id
        self._scene.post_status(f"Placed atom {atom.symbol}{atom.atom_id}.")

    def undo(self) -> None:
        if self._atom_id is None:
            return
        self._scene.remove_atom(self._atom_id)
        self._scene.post_status("Undid atom placement.")


class UpdateAtomElementCommand(QUndoCommand):
    """Change the element of an existing atom."""

    def __init__(self, scene: "MoleculeScene", atom_id: int, atomic_number: int) -> None:
        super().__init__("Change Atom Element")
        self._scene = scene
        self._atom_id = atom_id
        self._atomic_number = atomic_number
        atom = scene.document_snapshot().get_atom(atom_id)
        if atom is None:
            raise RuntimeError("Cannot edit an atom that does not exist.")
        self._previous_atom = atom
        self._updated_atom = atom.normalized()
        self._updated_atom.atomic_number = atomic_number

    def redo(self) -> None:
        atom = self._scene.update_atom_element(self._atom_id, self._atomic_number, select_updated=True)
        self._updated_atom = atom
        self._scene.post_status(f"Changed atom {atom.atom_id} to {atom.symbol}.")

    def undo(self) -> None:
        self._scene.replace_atom(self._previous_atom, select_updated=True)
        self._scene.post_status("Undid atom element change.")


class AddBondCommand(QUndoCommand):
    """Create a bond between two atoms, optionally creating the second atom."""

    def __init__(
        self,
        scene: "MoleculeScene",
        atom_a_id: int,
        bond_type: BondType,
        *,
        atom_b_id: int | None = None,
        atom_b_position: QPointF | None = None,
        atom_b_atomic_number: int | None = None,
    ) -> None:
        super().__init__("Add Bond")
        self._scene = scene
        self._atom_a_id = atom_a_id
        self._atom_b_id = atom_b_id
        self._bond_type = bond_type
        self._atom_b_position = QPointF(atom_b_position) if atom_b_position is not None else None
        self._atom_b_atomic_number = atom_b_atomic_number
        self._created_atom_id: int | None = None
        self._bond_id: int | None = None

    def redo(self) -> None:
        if self._atom_b_id is None:
            if self._atom_b_position is None or self._atom_b_atomic_number is None:
                raise RuntimeError("A target position and atomic number are required for a new bonded atom.")
            atom, bond = self._scene.add_bond_with_new_atom(
                atom_a_id=self._atom_a_id,
                atom_b_position=self._atom_b_position,
                atom_b_atomic_number=self._atom_b_atomic_number,
                bond_type=self._bond_type,
                preferred_atom_id=self._created_atom_id,
                preferred_bond_id=self._bond_id,
                select_created=True,
            )
            self._created_atom_id = atom.atom_id
        else:
            bond = self._scene.add_bond(
                atom_a_id=self._atom_a_id,
                atom_b_id=self._atom_b_id,
                bond_type=self._bond_type,
                preferred_bond_id=self._bond_id,
                select_created=True,
            )

        self._bond_id = bond.bond_id
        self._scene.post_status(
            f"Placed {bond.bond_type.display_name.lower()} bond {bond.bond_id} between atoms {bond.atom_a_id} and {bond.atom_b_id}."
        )

    def undo(self) -> None:
        if self._bond_id is not None:
            self._scene.remove_bond(self._bond_id)
        if self._created_atom_id is not None:
            self._scene.remove_atom(self._created_atom_id)
        self._scene.post_status("Undid bond placement.")


class UpdateBondTypeCommand(QUndoCommand):
    """Change the type of an existing bond."""

    def __init__(self, scene: "MoleculeScene", bond_id: int, bond_type: BondType) -> None:
        super().__init__("Change Bond Type")
        self._scene = scene
        self._bond_id = bond_id
        self._bond_type = bond_type
        bond = scene.document_snapshot().get_bond(bond_id)
        if bond is None:
            raise RuntimeError("Cannot edit a bond that does not exist.")
        self._previous_bond = bond

    def redo(self) -> None:
        bond = self._scene.update_bond_type(self._bond_id, self._bond_type, select_updated=True)
        self._scene.post_status(f"Changed bond {bond.bond_id} to {bond.bond_type.display_name.lower()}.")

    def undo(self) -> None:
        self._scene.replace_bond(self._previous_bond, select_updated=True)
        self._scene.post_status("Undid bond type change.")


class ReplaceDocumentCommand(QUndoCommand):
    """Replace the current document with another document snapshot."""

    def __init__(self, scene: "MoleculeScene", new_document: MoleculeDocument, text: str) -> None:
        super().__init__(text)
        self._scene = scene
        self._new_document = new_document.clone()
        self._previous_document = scene.document_snapshot()

    def redo(self) -> None:
        self._scene.load_document(self._new_document)
        self._scene.post_status(self.text())

    def undo(self) -> None:
        self._scene.load_document(self._previous_document)
        self._scene.post_status(f"Undid {self.text().lower()}.")


class DeleteSelectionCommand(QUndoCommand):
    """Delete a fragment of the current document."""

    def __init__(self, scene: "MoleculeScene", fragment: MoleculeDocument) -> None:
        super().__init__("Delete Selection")
        self._scene = scene
        self._fragment = fragment.clone()

    def redo(self) -> None:
        self._scene.delete_fragment(self._fragment)
        self._scene.post_status(
            f"Deleted {self._fragment.atom_count} atoms and {self._fragment.bond_count} bonds."
        )

    def undo(self) -> None:
        self._scene.restore_fragment(self._fragment)
        self._scene.post_status("Restored deleted selection.")


class ClearCanvasCommand(QUndoCommand):
    """Clear the editor scene while allowing undo."""

    def __init__(self, scene: "MoleculeScene", snapshot: MoleculeDocument) -> None:
        super().__init__("Clear Canvas")
        self._scene = scene
        self._snapshot = snapshot.clone()

    def redo(self) -> None:
        self._scene.load_document(MoleculeDocument())
        self._scene.post_status("Canvas cleared.")

    def undo(self) -> None:
        self._scene.load_document(self._snapshot)
        self._scene.post_status("Restored cleared canvas.")
