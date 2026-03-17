from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from chemstudio.data.models import Molecule, MoleculeDescriptor, MoleculeParameter


class MoleculeRepository:
    def __init__(self, session) -> None:
        self.session = session

    def upsert_molecule(self, payload: dict) -> Molecule:
        stmt = select(Molecule).where(Molecule.canonical_smiles == payload["canonical_smiles"])
        molecule = self.session.execute(stmt).scalar_one_or_none()

        if molecule is None:
            molecule = Molecule(
                name=payload["name"],
                canonical_smiles=payload["canonical_smiles"],
                inchi=payload["inchi"],
                inchikey=payload["inchikey"],
                molblock=payload["molblock"],
            )
            self.session.add(molecule)
            self.session.flush()
        else:
            molecule.name = payload["name"]
            molecule.inchi = payload["inchi"]
            molecule.inchikey = payload["inchikey"]
            molecule.molblock = payload["molblock"]

        parameters = {
            "input_smiles": payload["input_smiles"],
            "molecular_formula": payload["molecular_formula"],
            "molecular_weight": str(payload["molecular_weight"]),
        }
        self._replace_parameters(molecule, parameters)
        return molecule

    def _replace_parameters(self, molecule: Molecule, parameters: dict) -> None:
        existing = {item.key: item for item in molecule.parameters}
        for key, value in parameters.items():
            if key in existing:
                existing[key].value = str(value)
            else:
                molecule.parameters.append(MoleculeParameter(key=key, value=str(value)))

    def list_molecules(self) -> list[Molecule]:
        stmt = (
            select(Molecule)
            .options(joinedload(Molecule.parameters), joinedload(Molecule.descriptor_record))
            .order_by(Molecule.updated_at.desc())
        )
        return list(self.session.execute(stmt).unique().scalars().all())

    def get_molecule(self, molecule_id: int) -> Molecule | None:
        stmt = (
            select(Molecule)
            .where(Molecule.id == molecule_id)
            .options(joinedload(Molecule.parameters), joinedload(Molecule.descriptor_record))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def save_descriptors(self, molecule_id: int, descriptors: dict) -> MoleculeDescriptor:
        stmt = select(MoleculeDescriptor).where(MoleculeDescriptor.molecule_id == molecule_id)
        record = self.session.execute(stmt).scalar_one_or_none()
        if record is None:
            record = MoleculeDescriptor(molecule_id=molecule_id, descriptor_values=descriptors)
            self.session.add(record)
        else:
            record.descriptor_values = descriptors
        self.session.flush()
        return record

