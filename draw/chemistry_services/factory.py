"""Factory for chemistry service selection."""

from __future__ import annotations

from .base import ChemistryService, PlaceholderChemistryService
from .rdkit_adapter import RDKitChemistryService


def create_chemistry_service() -> ChemistryService:
    """Return the preferred chemistry backend for the current environment."""
    rdkit_service = RDKitChemistryService()
    if rdkit_service.available:
        return rdkit_service
    return PlaceholderChemistryService()

