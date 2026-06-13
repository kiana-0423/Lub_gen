"""Interfaces and fallback implementations for chemistry services."""

from __future__ import annotations

from typing import Protocol

from draw.core.models import BondType, MoleculeDocument


class ChemistryService(Protocol):
    """Common interface for chemistry backends."""

    name: str
    available: bool

    def describe(self) -> str:
        """Describe the backend status."""

    def import_smiles(self, smiles: str) -> MoleculeDocument:
        """Create a document from SMILES."""

    def export_smiles(self, document: MoleculeDocument) -> str:
        """Export a document as SMILES."""

    def sanitize(self, document: MoleculeDocument) -> MoleculeDocument:
        """Sanitize a document."""

    def refresh_document_state(self, document: MoleculeDocument) -> MoleculeDocument:
        """Refresh atom-level chemistry state for display and validation."""

    def generate_2d_coordinates(self, document: MoleculeDocument) -> MoleculeDocument:
        """Generate a fresh 2D layout for the document."""

    def export_mol(self, document: MoleculeDocument, molecule_name: str | None = None) -> str:
        """Export a document as MOL."""

    def export_sdf(self, document: MoleculeDocument, molecule_name: str | None = None) -> str:
        """Export a document as SDF."""

    def export_pdb(self, document: MoleculeDocument, molecule_name: str | None = None) -> str:
        """Export a document as PDB."""

    def expand_explicit_hydrogens(self, document: MoleculeDocument) -> MoleculeDocument:
        """Return a document with explicit hydrogens added."""


class ChemistryServiceError(RuntimeError):
    """Raised when chemistry import/export operations fail."""


class PlaceholderChemistryService:
    """Fallback used when RDKit is unavailable or advanced features are deferred."""

    name = "Placeholder chemistry service"
    available = False

    def describe(self) -> str:
        return "RDKit is not available yet. Chemistry import/export is disabled."

    def import_smiles(self, smiles: str) -> MoleculeDocument:
        raise self._unavailable()

    def export_smiles(self, document: MoleculeDocument) -> str:
        raise self._unavailable()

    def sanitize(self, document: MoleculeDocument) -> MoleculeDocument:
        return self.refresh_document_state(document)

    def refresh_document_state(self, document: MoleculeDocument) -> MoleculeDocument:
        refreshed = document.clone()
        aromatic_atom_ids = {
            atom_id
            for bond in refreshed.bonds.values()
            if bond.bond_type is BondType.AROMATIC
            for atom_id in (bond.atom_a_id, bond.atom_b_id)
        }
        for atom in refreshed.atoms.values():
            atom.implicit_hydrogens = 0
            atom.is_valid = True
            atom.validation_warning = None
            atom.aromatic = atom.atom_id in aromatic_atom_ids
        return refreshed

    def generate_2d_coordinates(self, document: MoleculeDocument) -> MoleculeDocument:
        raise self._unavailable()

    def export_mol(self, document: MoleculeDocument, molecule_name: str | None = None) -> str:
        raise self._unavailable()

    def export_sdf(self, document: MoleculeDocument, molecule_name: str | None = None) -> str:
        raise self._unavailable()

    def export_pdb(self, document: MoleculeDocument, molecule_name: str | None = None) -> str:
        raise self._unavailable()

    def expand_explicit_hydrogens(self, document: MoleculeDocument) -> MoleculeDocument:
        raise self._unavailable()

    @staticmethod
    def _unavailable() -> ChemistryServiceError:
        return ChemistryServiceError("RDKit is required for this chemistry operation.")
