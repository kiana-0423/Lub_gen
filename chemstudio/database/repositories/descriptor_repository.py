from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from chemstudio.database.db_manager import DatabaseManager


class DescriptorRepository:
    """Persistence facade for descriptors, feature names, and wide datasets."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def save_descriptors(
        self,
        molecule_id: int,
        descriptors: dict[str, object],
        fingerprint_bits: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.db_manager.connect() as connection:
            self._save_descriptors(connection, molecule_id, descriptors, now, fingerprint_bits=fingerprint_bits)
            connection.commit()

    def _save_descriptors(
        self,
        connection,
        molecule_id: int,
        descriptors: dict[str, object],
        timestamp: str,
        fingerprint_bits: str = "",
    ) -> None:
        existing = connection.execute(
            "SELECT id FROM molecule_descriptors WHERE molecule_id = ?",
            (molecule_id,),
        ).fetchone()
        payload = json.dumps(descriptors, ensure_ascii=False)
        if existing is None:
            connection.execute(
                """
                INSERT INTO molecule_descriptors (
                    molecule_id, descriptor_values_json, fingerprint_bits, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (molecule_id, payload, fingerprint_bits, timestamp, timestamp),
            )
        else:
            connection.execute(
                """
                UPDATE molecule_descriptors
                SET descriptor_values_json = ?, fingerprint_bits = ?, updated_at = ?
                WHERE molecule_id = ?
                """,
                (payload, fingerprint_bits, timestamp, molecule_id),
            )

    def list_feature_names(self) -> list[str]:
        with self.db_manager.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT feature_name FROM molecular_features ORDER BY feature_name"
            ).fetchall()
        return [str(row["feature_name"]) for row in rows]

    def list_property_names(self) -> list[str]:
        with self.db_manager.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT property_name FROM property_data ORDER BY property_name"
            ).fetchall()
        return [str(row["property_name"]) for row in rows]

    def get_wide_dataset(self, search_text: str = "", *, include_mordred: bool = False) -> pd.DataFrame:
        molecules = pd.DataFrame(self.db_manager.list_molecules(search_text=search_text))
        if molecules.empty:
            return pd.DataFrame(columns=["id", "name", "smiles", "source", "created_at"])

        with self.db_manager.connect() as connection:
            feature_rows = pd.read_sql_query(
                "SELECT molecule_id, feature_name, feature_value FROM molecular_features",
                connection,
            )
            property_rows = pd.read_sql_query(
                "SELECT molecule_id, property_name, property_value FROM property_data",
                connection,
            )
            descriptor_rows = pd.read_sql_query(
                "SELECT molecule_id, descriptor_values_json FROM molecule_descriptors",
                connection,
            ) if include_mordred else pd.DataFrame()

        dataset = molecules.copy()

        if not feature_rows.empty:
            feature_frame = (
                feature_rows.pivot_table(
                    index="molecule_id",
                    columns="feature_name",
                    values="feature_value",
                    aggfunc="mean",
                )
                .reset_index()
                .rename(columns={"molecule_id": "id"})
            )
            dataset = dataset.merge(feature_frame, on="id", how="left")

        if not property_rows.empty:
            property_frame = (
                property_rows.pivot_table(
                    index="molecule_id",
                    columns="property_name",
                    values="property_value",
                    aggfunc="mean",
                )
                .reset_index()
                .rename(columns={"molecule_id": "id"})
            )
            overlap = [column for column in property_frame.columns if column in dataset.columns and column != "id"]
            if overlap:
                property_frame = property_frame.rename(columns={column: f"property__{column}" for column in overlap})
            dataset = dataset.merge(property_frame, on="id", how="left")

        if include_mordred and not descriptor_rows.empty:
            descriptor_parts: list[dict[str, object]] = []
            for _, row in descriptor_rows.iterrows():
                descriptor_values = self._loads_json_dict(row["descriptor_values_json"])
                if not descriptor_values:
                    continue
                descriptor_parts.append({"id": int(row["molecule_id"]), **descriptor_values})
            if descriptor_parts:
                descriptor_frame = pd.DataFrame(descriptor_parts)
                overlap = [column for column in descriptor_frame.columns if column in dataset.columns and column != "id"]
                if overlap:
                    descriptor_frame = descriptor_frame.rename(
                        columns={column: f"descriptor__{column}" for column in overlap}
                    )
                dataset = dataset.merge(descriptor_frame, on="id", how="left")

        ordered_columns = ["id", "name", "smiles", "source", "created_at"]
        remaining_columns = [column for column in dataset.columns if column not in ordered_columns]
        return dataset[ordered_columns + sorted(remaining_columns)]

    def _loads_json_dict(self, raw_value: object) -> dict[str, object]:
        try:
            parsed = json.loads(str(raw_value or "{}"))
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
