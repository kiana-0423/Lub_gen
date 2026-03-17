from __future__ import annotations

from chemstudio.data.db import session_scope
from chemstudio.data.repositories.molecule_repository import MoleculeRepository

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, inchi
except ImportError:  # pragma: no cover
    Chem = None
    Descriptors = None
    rdMolDescriptors = None
    inchi = None


class MoleculeService:
    def __init__(self, repository: MoleculeRepository | None = None) -> None:
        self._repository = repository

    def _get_repository(self, session):
        return self._repository or MoleculeRepository(session)

    def validate_and_standardize(self, payload: dict) -> dict:
        smiles = (payload.get("smiles") or "").strip()
        if not smiles:
            raise ValueError("SMILES is required.")
        if Chem is None:
            raise ValueError("RDKit is not installed.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Unable to parse SMILES.")

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        result = {
            "name": (payload.get("name") or "").strip() or canonical_smiles,
            "input_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "molblock": payload.get("molblock") or "",
            "inchi": inchi.MolToInchi(mol) if inchi is not None else "",
            "inchikey": inchi.MolToInchiKey(mol) if inchi is not None else "",
            "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
            "molecular_weight": float(Descriptors.MolWt(mol)),
        }
        return result

    def save_molecule(self, payload: dict):
        standardized = self.validate_and_standardize(payload)
        with session_scope() as session:
            repository = self._get_repository(session)
            molecule = repository.upsert_molecule(standardized)
            session.flush()
            session.refresh(molecule)
            return molecule

    def list_molecules(self) -> list:
        with session_scope() as session:
            repository = self._get_repository(session)
            return repository.list_molecules()

    def get_molecule(self, molecule_id: int):
        with session_scope() as session:
            repository = self._get_repository(session)
            return repository.get_molecule(molecule_id)
