from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chemstudio.constants import FormulationRole


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
    material_type_id: int | None = None
    parameters: dict[str, object] = field(default_factory=dict)
    descriptors: dict[str, float] = field(default_factory=dict)
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
    material_type_id: int | None = None
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


@dataclass(slots=True)
class MaterialTypeRecord:
    id: int
    type_name: str
    category: str
    sub_category: str = ""
    description: str = ""
    typical_application: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class LubricantPropertyRecord:
    id: int
    molecule_id: int
    property_name: str
    property_value: float
    property_unit: str = ""
    test_standard: str = ""
    test_condition_json: str = "{}"
    is_blend_property: bool = False
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class FormulaComponentRecord:
    id: int
    formula_id: int
    molecule_id: int
    component_role: str = FormulationRole.ADDITIVE
    ratio: float = 0.0
    concentration: float | None = None
    concentration_unit: str = "wt%"
    sort_order: int = 0
    notes: str = ""


@dataclass(slots=True)
class AdditiveCompatibilityRecord:
    id: int
    additive_id: int
    base_oil_id: int
    compatibility_score: float | None = None
    solubility: str = ""
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(slots=True)
class FormulaTestResultRecord:
    id: int
    formula_id: int
    test_name: str
    test_standard: str = ""
    test_condition_json: str = "{}"
    result_value: float | None = None
    result_unit: str = ""
    is_predicted: bool = False
    model_id: int | None = None
    created_at: str = ""
