from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MoleculeImportRecord:
    """Structured molecule payload used by the import pipeline."""

    name: str
    smiles: str = ""
    source: str = ""
    features: dict[str, float] = field(default_factory=dict)
    properties: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class MoleculeDetail:
    """Detailed molecule record returned by the database layer."""

    id: int
    name: str
    smiles: str
    source: str
    created_at: str
    features: dict[str, float] = field(default_factory=dict)
    properties: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class FormulaRecord:
    """Saved formulation metadata."""

    id: int
    formula_name: str
    composition_json: str
    predicted_property_json: str
    created_at: str


@dataclass(slots=True)
class ModelArtifactSummary:
    """Serializable metadata for a trained model artifact."""

    model_name: str
    model_key: str
    target_name: str
    feature_names: list[str]
    metrics: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)
