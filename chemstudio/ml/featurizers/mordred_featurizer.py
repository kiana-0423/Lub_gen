from __future__ import annotations

import logging
import math

from rdkit import Chem
from mordred import Calculator, descriptors


logger = logging.getLogger(__name__)
_calculator = None


def _get_calculator():
    """Return a lazily initialized Mordred 2D descriptor calculator."""
    global _calculator
    if _calculator is not None:
        return _calculator
    _calculator = Calculator(descriptors, ignore_3D=True)
    logger.info("Initialized Mordred calculator with %d descriptors.", len(_calculator.descriptors))
    return _calculator


def compute_mordred_descriptors(smiles: str) -> dict[str, float]:
    """Compute valid numeric Mordred 2D descriptors for a SMILES string."""
    if not smiles.strip():
        return {}

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        logger.warning("RDKit could not parse SMILES for Mordred descriptors: %s", smiles)
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
