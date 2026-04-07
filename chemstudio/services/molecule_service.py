from __future__ import annotations

from collections.abc import Iterable, Mapping

from chemstudio.data.db import initialize_database, session_scope
from chemstudio.data.models import Molecule
from chemstudio.data.repositories.molecule_repository import MoleculeRepository

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, inchi, rdMolDescriptors
except ImportError:  # pragma: no cover
    Chem = None
    Descriptors = None
    inchi = None
    rdMolDescriptors = None


class MoleculeService:
    def __init__(self, repository_cls=MoleculeRepository, database_url: str | None = None) -> None:
        """配置仓储类型和数据库地址，供分子业务流程复用。"""
        self.repository_cls = repository_cls
        self.database_url = database_url

    def validate_and_standardize(self, payload: Mapping[str, object]) -> dict[str, object]:
        """校验分子输入并补全规范化后的结构化字段。"""
        smiles = str(payload.get("smiles") or payload.get("canonical_smiles") or "").strip()
        if not smiles:
            raise ValueError("SMILES is required.")

        if Chem is not None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Unable to parse SMILES.")

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            molecular_formula = rdMolDescriptors.CalcMolFormula(mol) if rdMolDescriptors else ""
            molecular_weight = float(Descriptors.MolWt(mol)) if Descriptors else 0.0
            inchi_value = inchi.MolToInchi(mol) if inchi is not None else ""
            inchikey_value = inchi.MolToInchiKey(mol) if inchi is not None else ""
        else:  # pragma: no cover
            canonical_smiles = smiles
            molecular_formula = ""
            molecular_weight = 0.0
            inchi_value = ""
            inchikey_value = ""

        user_parameters = payload.get("parameters") or {}
        if not isinstance(user_parameters, Mapping):
            raise ValueError("parameters must be a mapping.")

        parameters = {str(key): value for key, value in user_parameters.items()}
        parameters.update(
            {
                "input_smiles": smiles,
                "molecular_formula": molecular_formula,
                "molecular_weight": round(molecular_weight, 6),
            }
        )

        return {
            "code": payload.get("code"),
            "name": str(payload.get("name") or canonical_smiles).strip() or canonical_smiles,
            "input_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "inchi": inchi_value,
            "inchikey": inchikey_value,
            "molblock": payload.get("molblock") or "",
            "notes": payload.get("notes") or "",
            "is_hidden": bool(payload.get("is_hidden", False)),
            "parameters": parameters,
        }

    def save_molecule(self, payload: Mapping[str, object], molecule_id: int | None = None) -> dict[str, object]:
        """创建或更新分子记录，并返回序列化后的详情。"""
        initialize_database(self.database_url)
        standardized = self.validate_and_standardize(payload)
        with session_scope(self.database_url) as session:
            repository = self.repository_cls(session)
            molecule = repository.save_molecule(standardized, molecule_id=molecule_id)
            session.flush()
            session.refresh(molecule)
            molecule = repository.get_molecule(molecule.id)
            return self._serialize_molecule(molecule, include_detail=True)

    def import_molecules(self, records: Iterable[Mapping[str, object]]) -> dict[str, object]:
        """批量导入分子记录，并汇总返回保存结果。"""
        created_or_updated: list[dict[str, object]] = []
        for record in records:
            created_or_updated.append(self.save_molecule(record))
        return {"count": len(created_or_updated), "items": created_or_updated}

    def delete_molecule(self, molecule_id: int) -> bool:
        """删除指定分子记录。"""
        with session_scope(self.database_url) as session:
            repository = self.repository_cls(session)
            return repository.delete_molecule(molecule_id)

    def set_hidden_state(self, molecule_id: int, hidden: bool) -> dict[str, object] | None:
        """更新分子的隐藏状态，并返回更新后的详情。"""
        with session_scope(self.database_url) as session:
            repository = self.repository_cls(session)
            molecule = repository.set_hidden_state(molecule_id, hidden)
            if molecule is None:
                return None
            session.flush()
            session.refresh(molecule)
            molecule = repository.get_molecule(molecule.id)
            return self._serialize_molecule(molecule, include_detail=True)

    def get_molecule_detail(self, molecule_id: int) -> dict[str, object] | None:
        """读取单个分子的完整详情。"""
        with session_scope(self.database_url) as session:
            repository = self.repository_cls(session)
            molecule = repository.get_molecule(molecule_id)
            if molecule is None:
                return None
            return self._serialize_molecule(molecule, include_detail=True)

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
    ) -> dict[str, object]:
        """按分页、过滤和排序条件返回分子列表。"""
        with session_scope(self.database_url) as session:
            repository = self.repository_cls(session)
            items, total = repository.list_molecules(
                page=page,
                page_size=page_size,
                keyword=keyword,
                include_hidden=include_hidden,
                hidden_only=hidden_only,
                parameter_filters=parameter_filters,
                sort_by=sort_by,
                descending=descending,
            )
            return {
                "page": page,
                "page_size": page_size,
                "total": total,
                "items": [self._serialize_molecule(item, include_detail=False) for item in items],
            }

    def _serialize_molecule(self, molecule: Molecule | None, *, include_detail: bool) -> dict[str, object]:
        """把 ORM 分子对象转换成接口层稳定的字典结构。"""
        if molecule is None:
            raise ValueError("Molecule does not exist.")

        parameters = {item.key: item.value for item in molecule.parameters}
        descriptor_values = molecule.descriptor_record.descriptor_values if molecule.descriptor_record else {}
        data = {
            "id": molecule.id,
            "code": molecule.code,
            "name": molecule.name,
            "display_name": molecule.display_name,
            "input_smiles": molecule.input_smiles,
            "canonical_smiles": molecule.canonical_smiles,
            "inchi": molecule.inchi,
            "inchikey": molecule.inchikey,
            "is_hidden": molecule.is_hidden,
            "updated_at": molecule.updated_at.isoformat(),
            "created_at": molecule.created_at.isoformat(),
        }
        if include_detail:
            data.update(
                {
                    "molblock": molecule.molblock,
                    "notes": molecule.notes,
                    "parameters": parameters,
                    "descriptor_values": descriptor_values,
                }
            )
        return data
