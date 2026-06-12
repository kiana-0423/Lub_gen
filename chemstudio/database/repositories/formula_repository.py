from __future__ import annotations

import json
from datetime import datetime, timezone

from chemstudio.database.db_manager import DatabaseManager


class FormulaRepository:
    """Persistence facade for formulation records."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def save_formula(
        self,
        formula_name: str,
        composition: list[dict[str, object]],
        predicted_properties: dict[str, float],
        note: str = "",
        conditions: dict[str, float] | None = None,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        with self.db_manager.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO formulas (formula_name, note, composition_json, conditions_json, predicted_property_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    formula_name,
                    note,
                    json.dumps(composition, ensure_ascii=False, indent=2),
                    json.dumps(conditions or {}, ensure_ascii=False, indent=2),
                    json.dumps(predicted_properties, ensure_ascii=False, indent=2),
                    created_at,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def save_formulation(
        self,
        formula_name: str,
        note: str,
        composition: list[dict[str, object]],
        target_values: dict[str, float],
        conditions: dict[str, float] | None = None,
    ) -> int:
        return self.save_formula(
            formula_name=formula_name,
            note=note,
            composition=composition,
            conditions=conditions,
            predicted_properties=target_values,
        )

    def list_formulas(self, limit: int = 100) -> list[dict[str, object]]:
        with self.db_manager.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, formula_name, note, composition_json, conditions_json, predicted_property_json, created_at
                FROM formulas
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_formulations(self, limit: int = 200) -> list[dict[str, object]]:
        return self.list_formulas(limit=limit)

    def get_formulation(self, formulation_id: int) -> dict[str, object] | None:
        with self.db_manager.connect() as connection:
            row = connection.execute(
                """
                SELECT id, formula_name, note, composition_json, conditions_json, predicted_property_json, created_at
                FROM formulas
                WHERE id = ?
                """,
                (formulation_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def delete_formulation(self, formulation_id: int) -> bool:
        with self.db_manager.connect() as connection:
            cursor = connection.execute("DELETE FROM formulas WHERE id = ?", (formulation_id,))
            connection.commit()
            return cursor.rowcount > 0
