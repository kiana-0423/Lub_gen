from __future__ import annotations

from collections.abc import Mapping, Sequence

from chemstudio.database.db_manager import DatabaseManager


class MoleculeRepository:
    """SQLite-backed molecule repository facade."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def save_molecule(self, payload: Mapping[str, object], molecule_id: int | None = None) -> dict[str, object]:
        return self.db_manager.save_molecule(dict(payload), molecule_id=molecule_id)

    def delete_molecule(self, molecule_id: int) -> bool:
        return self.db_manager.delete_molecule(molecule_id)

    def set_hidden_state(self, molecule_id: int, hidden: bool) -> dict[str, object] | None:
        return self.db_manager.set_molecule_hidden_state(molecule_id, hidden)

    def get_molecule(self, molecule_id: int) -> dict[str, object] | None:
        detail = self.db_manager.get_molecule_detail(molecule_id)
        return self.db_manager.serialize_molecule_detail(detail) if detail is not None else None

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
        return self.db_manager.list_molecules_page(
            page=page,
            page_size=page_size,
            keyword=keyword,
            include_hidden=include_hidden,
            hidden_only=hidden_only,
            parameter_filters=dict(parameter_filters or {}),
            sort_by=sort_by,
            descending=descending,
        )

    def list_training_candidates(
        self,
        *,
        include_hidden: bool = False,
        molecule_ids: Sequence[int] | None = None,
        keyword: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, object]]:
        items = self.db_manager.list_molecules(
            keyword=keyword,
            include_hidden=include_hidden,
            sort_by="id",
            descending=False,
            limit=limit,
            offset=offset,
        )
        if molecule_ids:
            allowed = {int(molecule_id) for molecule_id in molecule_ids}
            items = [item for item in items if int(item["id"]) in allowed]
        return items

    def save_descriptors(self, molecule_id: int, descriptors: Mapping[str, object]) -> None:
        self.db_manager.save_descriptors(molecule_id, dict(descriptors))
