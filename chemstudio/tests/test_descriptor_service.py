from __future__ import annotations

import pytest

from chemstudio.services.descriptor_service import DescriptorService


@pytest.mark.skipif(DescriptorService.calculate.__globals__["Chem"] is None, reason="RDKit not installed")
def test_descriptor_service_calculates_basic_values():
    service = DescriptorService()
    result = service.calculate("CCO")
    assert "MolWt" in result
    assert result["HBA"] >= 1
