from __future__ import annotations

import pytest

from chemstudio.database.db_manager import DatabaseManager
from chemstudio.database.models import MoleculeImportRecord
from chemstudio.services.feature_service import FeatureService


@pytest.mark.skipif(FeatureService.compute_descriptors.__globals__["Chem"] is None, reason="RDKit not installed")
def test_feature_service_calculates_and_persists_basic_values(tmp_path):
    db_manager = DatabaseManager(tmp_path / "chemstudio.sqlite")
    db_manager.initialize_database()
    molecule_id = db_manager.insert_molecule_record(MoleculeImportRecord(name="ethanol", smiles="CCO"))
    service = FeatureService(db_manager)
    result = service.compute_and_persist_descriptors(molecule_id, "CCO")
    assert "mol_wt" in result
    assert result["h_acceptors"] >= 1
    detail = db_manager.get_molecule_detail(molecule_id)
    assert detail is not None
    assert "mol_wt" in detail.descriptor_values
