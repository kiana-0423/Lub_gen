from __future__ import annotations

try:  # pragma: no cover - optional dependency fallback
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


def validate_smiles(smiles: str) -> str:
    """Validate and canonicalize a SMILES string."""
    normalized = str(smiles or "").strip()
    if not normalized:
        raise ValueError("SMILES is required.")
    if Chem is None:  # pragma: no cover
        return normalized
    molecule = Chem.MolFromSmiles(normalized)
    if molecule is None:
        raise ValueError("Unable to parse SMILES.")
    return str(Chem.MolToSmiles(molecule, canonical=True))


def validate_molecule_name(name: str) -> str:
    """Validate a molecule name with a conservative length limit."""
    normalized = str(name or "").strip()
    if not normalized:
        raise ValueError("Molecule name is required.")
    if len(normalized) > 500:
        raise ValueError("Molecule name must be 500 characters or fewer.")
    return normalized


def validate_ratio(ratio: float) -> float:
    """Validate a formulation component ratio in the [0, 100] range."""
    value = float(ratio)
    if value < 0:
        raise ValueError("比例不能为负数。")
    if value > 100:
        raise ValueError("比例不能超过 100。")
    return value
