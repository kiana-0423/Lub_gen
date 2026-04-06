from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def normalize_field_name(value: Any) -> str:
    """Convert arbitrary column names into normalized snake_case keys."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def read_tabular_file(file_path: str | Path) -> pd.DataFrame:
    """Load CSV or Excel data into a dataframe."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def parse_feature_text(text: str) -> dict[str, float]:
    """Parse manual feature input from JSON or key=value lines."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return {}

    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        return {str(key): float(value) for key, value in payload.items()}

    features: dict[str, float] = {}
    separators = [",", "\n"]
    normalized_text = cleaned_text
    for separator in separators:
        normalized_text = normalized_text.replace(separator, "\n")

    for line in normalized_text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        features[key] = float(value)

    if not features:
        raise ValueError("Manual feature input must be JSON or use `key=value` pairs.")
    return features
