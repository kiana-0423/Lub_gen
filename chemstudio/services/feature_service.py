from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.utils.file_utils import normalize_field_name, parse_feature_text

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
except ImportError:  # pragma: no cover
    Chem = None
    Descriptors = None
    Lipinski = None
    rdMolDescriptors = None


class FeatureService:
    """Handles feature detection, descriptor generation, and input alignment."""

    METADATA_COLUMNS = {
        "id",
        "code",
        "name",
        "smiles",
        "input_smiles",
        "canonical_smiles",
        "inchi",
        "inchikey",
        "is_hidden",
        "source",
        "created_at",
        "updated_at",
    }

    def __init__(self, db_manager: DatabaseManager) -> None:
        """保存数据库访问依赖，用于推断训练特征列。"""
        self.db_manager = db_manager

    @property
    def rdkit_available(self) -> bool:
        """Whether RDKit is available in the current environment."""
        return Chem is not None

    def infer_feature_columns(self, dataset: pd.DataFrame, target_name: str) -> list[str]:
        """Identify numeric feature columns that should be used for model training."""
        property_names = set(self.db_manager.list_property_names())
        numeric_columns = list(dataset.select_dtypes(include=[np.number]).columns)
        feature_columns = [
            column
            for column in numeric_columns
            if column not in self.METADATA_COLUMNS and column != target_name and column not in property_names
        ]
        if feature_columns:
            return sorted(feature_columns)
        return [column for column in numeric_columns if column not in {"id", target_name}]

    def compute_descriptors(self, smiles: str) -> tuple[dict[str, float], str]:
        """Generate basic RDKit descriptors or return a compatibility message."""
        if not smiles.strip():
            return {}, "SMILES is empty."
        if not self.rdkit_available:
            return {}, "RDKit is not installed. Manual feature input will be used instead."

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return {}, "RDKit could not parse the SMILES string."

        descriptors = {
            "mol_wt": float(Descriptors.MolWt(molecule)),
            "mol_logp": float(Descriptors.MolLogP(molecule)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
            "h_donors": float(Lipinski.NumHDonors(molecule)),
            "h_acceptors": float(Lipinski.NumHAcceptors(molecule)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(molecule)),
            "ring_count": float(rdMolDescriptors.CalcNumRings(molecule)),
            "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
        }
        return descriptors, "RDKit descriptors generated successfully."

    def save_descriptors(self, molecule_id: int, descriptors: dict[str, object]) -> None:
        """Persist descriptor values for a molecule."""
        self.db_manager.save_descriptors(molecule_id, descriptors)

    def compute_and_persist_descriptors(self, molecule_id: int, smiles: str) -> dict[str, float]:
        """Compute descriptors and store them on the molecule detail record."""
        descriptors, message = self.compute_descriptors(smiles)
        if not descriptors:
            raise ValueError(message)
        self.save_descriptors(molecule_id, descriptors)
        return descriptors

    def build_prediction_features(
        self,
        smiles: str,
        feature_text: str,
        required_features: list[str],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Build an aligned feature vector for single-molecule prediction."""
        manual_features = {
            normalize_field_name(key): float(value)
            for key, value in parse_feature_text(feature_text).items()
        }
        descriptor_features, message = self.compute_descriptors(smiles)
        merged_features = {**manual_features, **descriptor_features}
        aligned_features = {feature_name: float(merged_features.get(feature_name, 0.0)) for feature_name in required_features}
        missing_features = [feature_name for feature_name in required_features if feature_name not in merged_features]
        return aligned_features, {
            "merged_features": merged_features,
            "missing_features": missing_features,
            "message": message,
        }
