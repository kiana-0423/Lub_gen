from __future__ import annotations

from chemstudio.database.db_manager import DatabaseManager


class MaterialRepository:
    """Persistence facade for lubricant material metadata."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def list_material_types(self, type_name: str | None = None) -> list[dict[str, object]]:
        return self.db_manager.list_material_types(type_name)

    def save_lubricant_property(
        self,
        molecule_id: int,
        property_name: str,
        property_value: float,
        **kwargs: object,
    ) -> int:
        return self.db_manager.save_lubricant_property(
            molecule_id=molecule_id,
            property_name=property_name,
            property_value=property_value,
            property_unit=str(kwargs.get("property_unit") or ""),
            test_standard=str(kwargs.get("test_standard") or ""),
            test_condition=kwargs.get("test_condition") if isinstance(kwargs.get("test_condition"), dict) else None,
            is_blend_property=bool(kwargs.get("is_blend_property", False)),
        )

    def get_lubricant_properties(self, molecule_id: int) -> list[dict[str, object]]:
        with self.db_manager.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, molecule_id, property_name, property_value, property_unit,
                       test_standard, test_condition_json, is_blend_property, created_at, updated_at
                FROM lubricant_properties
                WHERE molecule_id = ?
                ORDER BY property_name
                """,
                (molecule_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_additive_compatibility(self, additive_id: int, base_oil_id: int, **kwargs: object) -> int:
        score = kwargs.get("compatibility_score")
        return self.db_manager.save_additive_compatibility(
            additive_id=additive_id,
            base_oil_id=base_oil_id,
            compatibility_score=None if score is None else float(score),
            solubility=str(kwargs.get("solubility") or ""),
            notes=str(kwargs.get("notes") or ""),
        )

    def get_additive_compatibilities(self, **kwargs: object) -> list[dict[str, object]]:
        additive_id = kwargs.get("additive_id")
        base_oil_id = kwargs.get("base_oil_id")
        return self.db_manager.get_additive_compatibilities(
            additive_id=None if additive_id is None else int(additive_id),
            base_oil_id=None if base_oil_id is None else int(base_oil_id),
        )
