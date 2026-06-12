from __future__ import annotations

import json
from datetime import datetime, timezone

from chemstudio.database.db_manager import DatabaseManager


class ModelRepository:
    """Persistence facade for trained-model metadata."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def save_model_record(
        self,
        *,
        name: str,
        model_type: str,
        problem_type: str,
        target_name: str,
        feature_columns: list[str],
        metrics: dict[str, object],
        training_config: dict[str, object] | None = None,
        artifact_path: str = "",
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.db_manager.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO model_records (
                    name, model_type, problem_type, target_name, feature_columns_json,
                    metrics_json, training_config_json, artifact_path, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    model_type,
                    problem_type,
                    target_name,
                    json.dumps(feature_columns, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                    json.dumps(training_config or {}, ensure_ascii=False),
                    artifact_path,
                    now,
                    now,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def list_model_records(self, limit: int = 100) -> list[dict[str, object]]:
        with self.db_manager.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, name, model_type, problem_type, target_name, feature_columns_json,
                       metrics_json, training_config_json, artifact_path, created_at, updated_at
                FROM model_records
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
