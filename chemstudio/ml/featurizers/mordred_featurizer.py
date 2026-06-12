from __future__ import annotations

import logging
import math
from functools import lru_cache

from rdkit import Chem

try:  # pragma: no cover - optional in lightweight test environments
    from mordred import Calculator, descriptors
except ImportError:  # pragma: no cover
    Calculator = None
    descriptors = None


logger = logging.getLogger(__name__)


def is_mordred_available() -> bool:
    """Return whether Mordred descriptor calculation is available."""
    return Calculator is not None and descriptors is not None


@lru_cache(maxsize=1)
def _get_calculator():
    """Return a lazily initialized Mordred 2D descriptor calculator."""
    if not is_mordred_available():
        raise RuntimeError("Mordred is not installed.")
    calculator = Calculator(descriptors, ignore_3D=True)
    logger.info("Initialized Mordred calculator with %d descriptors.", len(calculator.descriptors))
    return calculator


def compute_mordred_descriptors(smiles: str) -> dict[str, float]:
    """Compute valid numeric Mordred 2D descriptors for a SMILES string."""
    if not smiles.strip():
        return {}

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        logger.warning("RDKit could not parse SMILES for Mordred descriptors: %s", smiles)
        return {}
    if not is_mordred_available():
        logger.warning("Mordred is not installed; skipping Mordred descriptors.")
        return {}

    calculator = _get_calculator()
    try:
        result = calculator(molecule)
    except Exception as exc:  # pragma: no cover - depends on Mordred internals
        logger.warning("Mordred descriptor calculation failed for %s: %s", smiles, exc)
        return {}

    values: dict[str, float] = {}
    for descriptor, raw_value in zip(calculator.descriptors, result):
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or math.isinf(value):
            continue
        values[str(descriptor)] = value
    return values


def get_descriptor_count(smiles: str) -> int:
    """Return the number of valid numeric Mordred descriptors for a SMILES string."""
    return len(compute_mordred_descriptors(smiles))
