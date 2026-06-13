"""Conversion helpers between editor documents and RDKit molecules."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from draw.core.models import Atom, Bond, BondType, MoleculeDocument

IMPORT_LAYOUT_SCALE = 50.0


@dataclass(slots=True)
class RDKitDocumentConversion:
    """A converted RDKit molecule plus atom and bond index mappings."""

    mol: Chem.Mol
    atom_id_to_rdkit_index: dict[int, int]
    rdkit_index_to_atom_id: dict[int, int]
    bond_id_to_rdkit_index: dict[int, int]
    rdkit_index_to_bond_id: dict[int, int]


def document_to_rdkit_conversion(
    document: MoleculeDocument,
    *,
    sanitize: bool = True,
    include_conformer: bool = True,
) -> RDKitDocumentConversion:
    """Convert an editor document into an RDKit molecule plus index mappings."""
    if document.atom_count == 0:
        raise ValueError("Cannot convert an empty document.")

    rw_mol = Chem.RWMol()
    atom_id_to_rdkit_index: dict[int, int] = {}
    rdkit_index_to_atom_id: dict[int, int] = {}
    bond_id_to_rdkit_index: dict[int, int] = {}
    rdkit_index_to_bond_id: dict[int, int] = {}

    for atom in sorted(document.atoms.values(), key=lambda item: item.atom_id):
        rd_atom = Chem.Atom(atom.atomic_number)
        rd_atom.SetFormalCharge(atom.formal_charge)
        rd_atom.SetIsAromatic(atom.aromatic)
        rdkit_index = rw_mol.AddAtom(rd_atom)
        atom_id_to_rdkit_index[atom.atom_id] = rdkit_index
        rdkit_index_to_atom_id[rdkit_index] = atom.atom_id

    for bond in sorted(document.bonds.values(), key=lambda item: item.bond_id):
        atom_a_index = atom_id_to_rdkit_index.get(bond.atom_a_id)
        atom_b_index = atom_id_to_rdkit_index.get(bond.atom_b_id)
        if atom_a_index is None or atom_b_index is None:
            raise ValueError("Bond references an atom that does not exist in the document.")

        rw_mol.AddBond(atom_a_index, atom_b_index, _rdkit_bond_type_for_document_bond(bond.bond_type))
        rd_bond = rw_mol.GetBondBetweenAtoms(atom_a_index, atom_b_index)
        if rd_bond is None:
            raise ValueError("Failed to create bond in the RDKit molecule.")

        if bond.bond_type is BondType.AROMATIC:
            rd_bond.SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(atom_a_index).SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(atom_b_index).SetIsAromatic(True)

        bond_id_to_rdkit_index[bond.bond_id] = rd_bond.GetIdx()
        rdkit_index_to_bond_id[rd_bond.GetIdx()] = bond.bond_id

    mol = rw_mol.GetMol()

    if include_conformer:
        conformer = Chem.Conformer(mol.GetNumAtoms())
        conformer.Set3D(False)
        for atom in sorted(document.atoms.values(), key=lambda item: item.atom_id):
            rdkit_index = atom_id_to_rdkit_index[atom.atom_id]
            conformer.SetAtomPosition(rdkit_index, Point3D(float(atom.x), float(atom.y), 0.0))
        mol.RemoveAllConformers()
        mol.AddConformer(conformer, assignId=True)

    if sanitize:
        Chem.SanitizeMol(mol)

    return RDKitDocumentConversion(
        mol=mol,
        atom_id_to_rdkit_index=atom_id_to_rdkit_index,
        rdkit_index_to_atom_id=rdkit_index_to_atom_id,
        bond_id_to_rdkit_index=bond_id_to_rdkit_index,
        rdkit_index_to_bond_id=rdkit_index_to_bond_id,
    )


def document_to_rdkit_mol(
    document: MoleculeDocument,
    *,
    sanitize: bool = True,
    include_conformer: bool = True,
) -> Chem.Mol:
    """Convert an editor document into an RDKit molecule."""
    return document_to_rdkit_conversion(
        document,
        sanitize=sanitize,
        include_conformer=include_conformer,
    ).mol


def rdkit_mol_to_document(
    mol: Chem.Mol,
    *,
    generate_2d_if_missing: bool = True,
    scale: float = 1.0,
    center: bool = False,
    invert_y: bool = False,
) -> MoleculeDocument:
    """Convert an RDKit molecule into an editor document."""
    working_mol = Chem.Mol(mol)
    if working_mol.GetNumAtoms() == 0:
        return MoleculeDocument()

    if generate_2d_if_missing and working_mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(working_mol)

    conformer = working_mol.GetConformer() if working_mol.GetNumConformers() else None
    center_x = 0.0
    center_y = 0.0

    if conformer is not None and center:
        atom_count = working_mol.GetNumAtoms()
        center_x = sum(conformer.GetAtomPosition(index).x for index in range(atom_count)) / atom_count
        center_y = sum(conformer.GetAtomPosition(index).y for index in range(atom_count)) / atom_count

    document = MoleculeDocument()

    for atom_index, atom in enumerate(working_mol.GetAtoms(), start=1):
        x_pos = 0.0
        y_pos = 0.0
        if conformer is not None:
            position = conformer.GetAtomPosition(atom_index - 1)
            x_pos = (position.x - center_x) * scale
            y_pos = (position.y - center_y) * scale
            if invert_y:
                y_pos *= -1

        document.add_atom(
            Atom(
                atom_id=atom_index,
                atomic_number=atom.GetAtomicNum(),
                symbol=atom.GetSymbol(),
                x=x_pos,
                y=y_pos,
                formal_charge=atom.GetFormalCharge(),
                aromatic=atom.GetIsAromatic(),
                implicit_hydrogens=atom.GetNumImplicitHs(),
            )
        )

    for bond_index, bond in enumerate(working_mol.GetBonds(), start=1):
        document.add_bond(
            Bond(
                bond_id=bond_index,
                atom_a_id=bond.GetBeginAtomIdx() + 1,
                atom_b_id=bond.GetEndAtomIdx() + 1,
                bond_type=_document_bond_type_for_rdkit_bond(bond),
            )
        )

    return document


def imported_rdkit_mol_to_document(mol: Chem.Mol) -> MoleculeDocument:
    """Convert an imported RDKit molecule into centered editor coordinates."""
    return rdkit_mol_to_document(
        mol,
        generate_2d_if_missing=True,
        scale=IMPORT_LAYOUT_SCALE,
        center=True,
        invert_y=True,
    )


def _rdkit_bond_type_for_document_bond(bond_type: BondType) -> Chem.BondType:
    if bond_type is BondType.DOUBLE:
        return Chem.BondType.DOUBLE
    if bond_type is BondType.TRIPLE:
        return Chem.BondType.TRIPLE
    if bond_type is BondType.AROMATIC:
        return Chem.BondType.AROMATIC
    return Chem.BondType.SINGLE


def _document_bond_type_for_rdkit_bond(bond: Chem.Bond) -> BondType:
    if bond.GetIsAromatic() or bond.GetBondType() is Chem.BondType.AROMATIC:
        return BondType.AROMATIC
    if bond.GetBondType() is Chem.BondType.DOUBLE:
        return BondType.DOUBLE
    if bond.GetBondType() is Chem.BondType.TRIPLE:
        return BondType.TRIPLE
    return BondType.SINGLE
