from __future__ import annotations

from collections.abc import Mapping, Sequence

from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.orm import joinedload

from chemstudio.data.models import Molecule, MoleculeDescriptor, MoleculeParameter


class MoleculeRepository:
    def __init__(self, session) -> None:
        self.session = session

    def save_molecule(self, payload: Mapping[str, object], molecule_id: int | None = None) -> Molecule:
        molecule = self._resolve_target_molecule(payload, molecule_id)

        if molecule is None:
            molecule = Molecule()
            self.session.add(molecule)

        molecule.code = self._clean_nullable(payload.get("code"))
        molecule.name = str(payload["name"]).strip()
        molecule.input_smiles = str(payload.get("input_smiles") or "")
        molecule.canonical_smiles = str(payload["canonical_smiles"]).strip()
        molecule.inchi = str(payload.get("inchi") or "")
        molecule.inchikey = str(payload.get("inchikey") or "")
        molecule.molblock = str(payload.get("molblock") or "")
        molecule.notes = str(payload.get("notes") or "")
        molecule.is_hidden = bool(payload.get("is_hidden", False))
        self.session.flush()

        parameters = payload.get("parameters") or {}
        if not isinstance(parameters, Mapping):
            raise ValueError("parameters must be a mapping.")
        self._replace_parameters(molecule, parameters)
        return molecule

    def delete_molecule(self, molecule_id: int) -> bool:
        molecule = self.get_molecule(molecule_id)
        if molecule is None:
            return False
        self.session.delete(molecule)
        return True

    def set_hidden_state(self, molecule_id: int, hidden: bool) -> Molecule | None:
        molecule = self.get_molecule(molecule_id)
        if molecule is None:
            return None
        molecule.is_hidden = hidden
        self.session.flush()
        return molecule

    def get_molecule(self, molecule_id: int) -> Molecule | None:
        stmt = (
            select(Molecule)
            .where(Molecule.id == molecule_id)
            .options(joinedload(Molecule.parameters), joinedload(Molecule.descriptor_record))
        )
        return self.session.execute(stmt).unique().scalar_one_or_none()

    def list_molecules(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        keyword: str | None = None,
        include_hidden: bool = False,
        hidden_only: bool = False,
        parameter_filters: Mapping[str, object] | None = None,
        sort_by: str = "updated_at",
        descending: bool = True,
    ) -> tuple[list[Molecule], int]:
        filters = self._build_filters(
            keyword=keyword,
            include_hidden=include_hidden,
            hidden_only=hidden_only,
            parameter_filters=parameter_filters,
        )
        sort_column = self._resolve_sort_column(sort_by)
        order_by = sort_column.desc() if descending else sort_column.asc()

        stmt = (
            select(Molecule)
            .where(*filters)
            .options(joinedload(Molecule.parameters), joinedload(Molecule.descriptor_record))
            .order_by(order_by, Molecule.id.desc())
            .offset(max(page - 1, 0) * page_size)
            .limit(page_size)
        )
        count_stmt = select(func.count(Molecule.id)).where(*filters)

        items = list(self.session.execute(stmt).unique().scalars().all())
        total = int(self.session.execute(count_stmt).scalar_one())
        return items, total

    def list_training_candidates(
        self,
        *,
        include_hidden: bool = False,
        molecule_ids: Sequence[int] | None = None,
        keyword: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Molecule]:
        filters = self._build_filters(
            keyword=keyword,
            include_hidden=include_hidden,
            hidden_only=False,
            parameter_filters=None,
        )
        if molecule_ids:
            filters.append(Molecule.id.in_(list(molecule_ids)))

        stmt = (
            select(Molecule)
            .where(*filters)
            .options(joinedload(Molecule.parameters), joinedload(Molecule.descriptor_record))
            .order_by(Molecule.id.asc())
            .offset(max(offset, 0))
        )
        if limit is not None:
            stmt = stmt.limit(limit)

        return list(self.session.execute(stmt).unique().scalars().all())

    def save_descriptors(self, molecule_id: int, descriptors: Mapping[str, object]) -> MoleculeDescriptor:
        stmt = select(MoleculeDescriptor).where(MoleculeDescriptor.molecule_id == molecule_id)
        record = self.session.execute(stmt).scalar_one_or_none()
        if record is None:
            record = MoleculeDescriptor(molecule_id=molecule_id, descriptor_values=dict(descriptors))
            self.session.add(record)
        else:
            record.descriptor_values = dict(descriptors)
        self.session.flush()
        return record

    def _resolve_target_molecule(self, payload: Mapping[str, object], molecule_id: int | None) -> Molecule | None:
        if molecule_id is not None:
            molecule = self.get_molecule(molecule_id)
            if molecule is None:
                raise ValueError(f"Molecule {molecule_id} does not exist.")
            return molecule

        code = self._clean_nullable(payload.get("code"))
        if code:
            stmt = select(Molecule).where(Molecule.code == code)
            molecule = self.session.execute(stmt).scalar_one_or_none()
            if molecule is not None:
                return molecule

        canonical_smiles = str(payload.get("canonical_smiles") or "").strip()
        if canonical_smiles:
            stmt = select(Molecule).where(Molecule.canonical_smiles == canonical_smiles)
            return self.session.execute(stmt).scalar_one_or_none()

        return None

    def _replace_parameters(self, molecule: Molecule, parameters: Mapping[str, object]) -> None:
        existing = {item.key: item for item in molecule.parameters}
        normalized = {str(key): "" if value is None else str(value) for key, value in parameters.items()}

        for key, item in list(existing.items()):
            if key not in normalized:
                molecule.parameters.remove(item)

        for key, value in normalized.items():
            item = existing.get(key)
            if item is None:
                molecule.parameters.append(MoleculeParameter(key=key, value=value))
            else:
                item.value = value

    def _build_filters(
        self,
        *,
        keyword: str | None,
        include_hidden: bool,
        hidden_only: bool,
        parameter_filters: Mapping[str, object] | None,
    ) -> list[object]:
        filters: list[object] = []

        if hidden_only:
            filters.append(Molecule.is_hidden.is_(True))
        elif not include_hidden:
            filters.append(Molecule.is_hidden.is_(False))

        if keyword:
            pattern = f"%{keyword.strip()}%"
            parameter_match = exists(
                select(MoleculeParameter.id).where(
                    and_(
                        MoleculeParameter.molecule_id == Molecule.id,
                        or_(
                            MoleculeParameter.key.ilike(pattern),
                            MoleculeParameter.value.ilike(pattern),
                        ),
                    )
                )
            )
            filters.append(
                or_(
                    Molecule.name.ilike(pattern),
                    Molecule.code.ilike(pattern),
                    Molecule.canonical_smiles.ilike(pattern),
                    Molecule.input_smiles.ilike(pattern),
                    Molecule.inchikey.ilike(pattern),
                    parameter_match,
                )
            )

        for key, value in (parameter_filters or {}).items():
            filters.append(
                exists(
                    select(MoleculeParameter.id).where(
                        and_(
                            MoleculeParameter.molecule_id == Molecule.id,
                            MoleculeParameter.key == str(key),
                            MoleculeParameter.value == str(value),
                        )
                    )
                )
            )

        return filters

    def _resolve_sort_column(self, sort_by: str):
        allowed = {
            "id": Molecule.id,
            "name": Molecule.name,
            "code": Molecule.code,
            "canonical_smiles": Molecule.canonical_smiles,
            "created_at": Molecule.created_at,
            "updated_at": Molecule.updated_at,
        }
        return allowed.get(sort_by, Molecule.updated_at)

    @staticmethod
    def _clean_nullable(value: object) -> str | None:
        if value is None:
            return None
        result = str(value).strip()
        return result or None
