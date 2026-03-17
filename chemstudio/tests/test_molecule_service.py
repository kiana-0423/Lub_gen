from __future__ import annotations

import pytest

from chemstudio.data.db import initialize_database
from chemstudio.services.molecule_service import MoleculeService


@pytest.mark.skipif(MoleculeService().validate_and_standardize.__globals__["Chem"] is None, reason="RDKit not installed")
def test_molecule_service_standardizes_smiles():
    initialize_database()
    service = MoleculeService()
    result = service.validate_and_standardize({"smiles": "C(CO)O", "name": "glycol"})
    assert result["canonical_smiles"]
    assert result["molecular_formula"] == "C2H6O2"

