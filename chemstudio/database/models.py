from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MoleculeImportRecord:
    """Structured molecule payload used by the import pipeline."""

    name: str
    smiles: str = ""
    source: str = ""
    code: str | None = None
    input_smiles: str = ""
    canonical_smiles: str = ""
    inchi: str = ""
    inchikey: str = ""
    molblock: str = ""
    notes: str = ""
    is_hidden: bool = False
    parameters: dict[str, object] = field(default_factory=dict)
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
    code: str | None = None
    input_smiles: str = ""
    canonical_smiles: str = ""
    inchi: str = ""
    inchikey: str = ""
    molblock: str = ""
    notes: str = ""
    is_hidden: bool = False
    updated_at: str = ""
    parameters: dict[str, object] = field(default_factory=dict)
    descriptor_values: dict[str, object] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    properties: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class FormulaRecord:
    """Saved formulation metadata."""

    id: int
    formula_name: str
    note: str
    composition_json: str
    conditions_json: str
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


@dataclass(slots=True)
class ModelRecord:
    """Persisted trained-model metadata."""

    id: int
    name: str
    model_type: str
    problem_type: str
    target_name: str
    feature_columns_json: str
    metrics_json: str
    training_config_json: str
    artifact_path: str
    created_at: str
    updated_at: str
