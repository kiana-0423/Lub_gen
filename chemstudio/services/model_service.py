from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class ModelService:
    def train_regressor(self, dataframe: pd.DataFrame, target_column: str) -> tuple[dict, object]:
        features = dataframe.drop(columns=[target_column])
        target = dataframe[target_column]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(features, target)
        metrics = {"rows": len(dataframe), "features": features.shape[1]}
        return metrics, model

    def save_metadata(self, model_dir: Path, metadata: dict) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        output = model_dir / "metadata.json"
        output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return output

