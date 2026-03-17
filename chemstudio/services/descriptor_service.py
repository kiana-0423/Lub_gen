from __future__ import annotations

from chemstudio.data.db import session_scope
from chemstudio.data.repositories.molecule_repository import MoleculeRepository

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
except ImportError:  # pragma: no cover
    Chem = None
    Crippen = None
    Descriptors = None
    Lipinski = None
    rdMolDescriptors = None


class DescriptorService:
    def calculate(self, smiles: str) -> dict:
        if Chem is None:
            raise ValueError("RDKit is not installed.")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Unable to parse SMILES for descriptor calculation.")

        return {
            "MolWt": round(float(Descriptors.MolWt(mol)), 4),
            "LogP": round(float(Crippen.MolLogP(mol)), 4),
            "TPSA": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
            "HBD": int(Lipinski.NumHDonors(mol)),
            "HBA": int(Lipinski.NumHAcceptors(mol)),
            "RotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
            "RingCount": int(rdMolDescriptors.CalcNumRings(mol)),
        }

    def calculate_and_persist(self, molecule_id: int, smiles: str) -> dict:
        descriptors = self.calculate(smiles)
        with session_scope() as session:
            repository = MoleculeRepository(session)
            repository.save_descriptors(molecule_id, descriptors)
        return descriptors

