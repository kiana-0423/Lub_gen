"""RDKit-backed chemistry service implementation."""

from __future__ import annotations

from typing import Final

from draw.core.models import BondType, MoleculeDocument

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from .conversion import (
        document_to_rdkit_conversion,
        document_to_rdkit_mol,
        imported_rdkit_mol_to_document,
        rdkit_mol_to_document,
    )
except ImportError:  # pragma: no cover - depends on local environment
    Chem = None
    AllChem = None

from .base import ChemistryServiceError


class RDKitChemistryService:
    """RDKit-backed chemistry adapter for import/export and validation tasks."""

    DEFAULT_MOLECULE_NAME: Final[str] = "ChemDrawingSmiles"

    name = "RDKit"

    def __init__(self) -> None:
        self.available = Chem is not None and AllChem is not None

    def describe(self) -> str:
        if self.available:
            return "RDKit detected. Validation, implicit hydrogens, and structure export are available."
        return "RDKit is not installed. The UI can run, but chemistry features remain unavailable."

    def import_smiles(self, smiles: str) -> MoleculeDocument:
        self._ensure_available()
        normalized_smiles = smiles.strip()
        if not normalized_smiles:
            raise ChemistryServiceError("SMILES input is empty.")

        mol = Chem.MolFromSmiles(normalized_smiles, sanitize=False)
        if mol is None:
            raise ChemistryServiceError("RDKit could not parse the SMILES string.")

        self._sanitize_mol(mol, context="SMILES import")
        AllChem.Compute2DCoords(mol)
        return self.refresh_document_state(imported_rdkit_mol_to_document(mol))

    def export_smiles(self, document: MoleculeDocument) -> str:
        mol = self._document_to_sanitized_mol(document)
        return Chem.MolToSmiles(mol, canonical=True)

    def sanitize(self, document: MoleculeDocument) -> MoleculeDocument:
        mol = self._document_to_sanitized_mol(document)
        return self.refresh_document_state(
            rdkit_mol_to_document(
                mol,
                generate_2d_if_missing=False,
                scale=1.0,
                center=False,
                invert_y=False,
            )
        )

    def refresh_document_state(self, document: MoleculeDocument) -> MoleculeDocument:
        self._ensure_available()
        refreshed = document.clone()
        if refreshed.atom_count == 0:
            return refreshed

        try:
            conversion = document_to_rdkit_conversion(refreshed, sanitize=False, include_conformer=True)
        except ValueError:
            return refreshed
        except Exception as exc:
            raise ChemistryServiceError(f"RDKit could not analyze the current drawing: {exc}") from exc

        mol = conversion.mol
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass

        warning_map: dict[int, list[str]] = {}
        for problem in Chem.DetectChemistryProblems(mol):
            if hasattr(problem, "GetAtomIdx"):
                atom_idx = problem.GetAtomIdx()
                atom_id = conversion.rdkit_index_to_atom_id.get(atom_idx)
                if atom_id is not None:
                    warning_map.setdefault(atom_id, []).append(problem.GetType())

        sanitized = False
        try:
            Chem.SanitizeMol(mol)
            sanitized = True
        except Exception:
            sanitized = False

        aromatic_atom_ids: set[int] = set()
        if sanitized:
            for atom_idx, rd_atom in enumerate(mol.GetAtoms()):
                atom_id = conversion.rdkit_index_to_atom_id[atom_idx]
                atom = refreshed.atoms[atom_id]
                atom.implicit_hydrogens = rd_atom.GetNumImplicitHs()
                atom.aromatic = rd_atom.GetIsAromatic()
                atom.is_valid = atom_id not in warning_map
                atom.validation_warning = "; ".join(warning_map.get(atom_id, [])) or None
                if atom.aromatic:
                    aromatic_atom_ids.add(atom_id)

            for bond in refreshed.bonds.values():
                rdkit_index = conversion.bond_id_to_rdkit_index.get(bond.bond_id)
                if rdkit_index is None:
                    continue
                rd_bond = mol.GetBondWithIdx(rdkit_index)
                if rd_bond.GetIsAromatic():
                    bond.bond_type = BondType.AROMATIC
                    aromatic_atom_ids.update((bond.atom_a_id, bond.atom_b_id))
        else:
            aromatic_atom_ids = {
                atom_id
                for bond in refreshed.bonds.values()
                if bond.bond_type is BondType.AROMATIC
                for atom_id in (bond.atom_a_id, bond.atom_b_id)
            }
            for atom_id, atom in refreshed.atoms.items():
                atom.implicit_hydrogens = 0
                atom.aromatic = atom_id in aromatic_atom_ids
                warnings = warning_map.get(atom_id)
                atom.is_valid = warnings is None
                atom.validation_warning = "; ".join(warnings) if warnings else None

        return refreshed

    def generate_2d_coordinates(self, document: MoleculeDocument) -> MoleculeDocument:
        mol = self._document_to_sanitized_mol(document)
        mol.RemoveAllConformers()
        AllChem.Compute2DCoords(mol)
        return self.refresh_document_state(imported_rdkit_mol_to_document(mol))

    def export_mol(self, document: MoleculeDocument, molecule_name: str | None = None) -> str:
        mol = self._document_to_sanitized_mol(document, molecule_name=molecule_name)
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
        return Chem.MolToMolBlock(mol)

    def export_sdf(self, document: MoleculeDocument, molecule_name: str | None = None) -> str:
        mol_block = self.export_mol(document, molecule_name=molecule_name).rstrip()
        return f"{mol_block}\n$$$$\n"

    def export_pdb(self, document: MoleculeDocument, molecule_name: str | None = None) -> str:
        mol = self._document_to_sanitized_mol(document, molecule_name=molecule_name)
        pdb_mol = Chem.AddHs(Chem.Mol(mol), addCoords=True)
        pdb_mol.SetProp("_Name", molecule_name or self.DEFAULT_MOLECULE_NAME)

        if pdb_mol.GetNumAtoms() > 0:
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xF00D
            status = AllChem.EmbedMolecule(pdb_mol, params)
            if status != 0:
                params.useRandomCoords = True
                status = AllChem.EmbedMolecule(pdb_mol, params)

            if status == 0:
                try:
                    AllChem.UFFOptimizeMolecule(pdb_mol, maxIters=200)
                except Exception:
                    pass
            else:
                AllChem.Compute2DCoords(pdb_mol)

        return Chem.MolToPDBBlock(pdb_mol)

    def expand_explicit_hydrogens(self, document: MoleculeDocument) -> MoleculeDocument:
        mol = self._document_to_sanitized_mol(document)
        expanded = Chem.AddHs(Chem.Mol(mol), addCoords=True)
        if expanded.GetNumConformers() == 0:
            AllChem.Compute2DCoords(expanded)
        return self.refresh_document_state(
            rdkit_mol_to_document(
                expanded,
                generate_2d_if_missing=False,
                scale=1.0,
                center=False,
                invert_y=False,
            )
        )

    def _document_to_sanitized_mol(
        self,
        document: MoleculeDocument,
        *,
        molecule_name: str | None = None,
    ):
        self._ensure_available()
        try:
            mol = document_to_rdkit_mol(document, sanitize=True, include_conformer=True)
        except ValueError as exc:
            raise ChemistryServiceError(str(exc)) from exc
        except Exception as exc:
            raise ChemistryServiceError(f"RDKit could not convert the current drawing: {exc}") from exc

        mol.SetProp("_Name", molecule_name or self.DEFAULT_MOLECULE_NAME)
        return mol

    def _sanitize_mol(self, mol, *, context: str) -> None:
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ChemistryServiceError(f"RDKit could not sanitize the molecule during {context}: {exc}") from exc

    def _ensure_available(self) -> None:
        if not self.available:
            raise ChemistryServiceError("RDKit is required for this chemistry operation.")
