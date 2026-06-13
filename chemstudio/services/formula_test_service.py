from __future__ import annotations

from typing import Any

from chemstudio.database.db_manager import DatabaseManager


class FormulaTestService:
    """Service facade for formulation-level test results."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def save_test_result(
        self,
        *,
        formula_id: int,
        test_name: str,
        result_value: float | None,
        test_standard: str = "",
        test_condition: dict[str, object] | None = None,
        result_unit: str = "",
        is_predicted: bool = False,
        model_id: int | None = None,
    ) -> int:
        return self.db_manager.save_formula_test_result(
            formula_id=formula_id,
            test_name=test_name,
            result_value=result_value,
            test_standard=test_standard,
            test_condition=test_condition,
            result_unit=result_unit,
            is_predicted=is_predicted,
            model_id=model_id,
        )

    def get_test_results(self, formula_id: int) -> list[dict[str, Any]]:
        return self.db_manager.get_formula_test_results(formula_id)
