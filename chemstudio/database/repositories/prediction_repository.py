from __future__ import annotations

import json
from datetime import datetime, timezone

from chemstudio.database.db_manager import DatabaseManager


class PredictionRepository:
    """Persistence facade for prediction records."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def save_prediction_record(
        self,
        *,
        model_id: int,
        molecule_id: int | None = None,
        predicted_value: float | None = None,
        predicted_label: str = "",
        confidence: float | None = None,
        input_features: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.db_manager.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions (
                    model_id, molecule_id, predicted_value, predicted_label, confidence,
                    input_features_json, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    molecule_id,
                    predicted_value,
                    predicted_label,
                    confidence,
                    json.dumps(input_features or {}, ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)
