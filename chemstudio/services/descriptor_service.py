from __future__ import annotations

from collections.abc import Mapping

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
    def calculate(self, smiles: str) -> dict[str, float | int]:
        """基于 SMILES 计算一组常用的 RDKit 描述符。"""
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

    def calculate_and_persist(
        self,
        molecule_id: int,
        smiles: str,
        *,
        database_url: str | None = None,
    ) -> dict[str, float | int]:
        """计算描述符并写回数据库中的分子描述符记录。"""
        descriptors = self.calculate(smiles)
        with session_scope(database_url) as session:
            repository = MoleculeRepository(session)
            repository.save_descriptors(molecule_id, descriptors)
        return descriptors

    def ensure_descriptor_values(
        self,
        molecule_id: int,
        smiles: str,
        descriptor_record: Mapping[str, object] | None = None,
        *,
        database_url: str | None = None,
    ) -> dict[str, float | int]:
        """优先复用已有描述符记录，缺失时再触发计算和持久化。"""
        if descriptor_record:
            return dict(descriptor_record)
        return self.calculate_and_persist(molecule_id, smiles, database_url=database_url)
