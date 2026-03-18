from __future__ import annotations

import pytest

from chemstudio.services.molecule_service import MoleculeService


@pytest.mark.skipif(MoleculeService.validate_and_standardize.__globals__["Chem"] is None, reason="RDKit not installed")
def test_molecule_crud_search_and_hidden_filters(chemstudio_env):
    service = MoleculeService()

    glycol = service.save_molecule(
        {
            "code": "M-001",
            "name": "glycol",
            "smiles": "C(CO)O",
            "parameters": {"family": "solvent", "viscosity": 12.5},
        }
    )
    ethanol = service.save_molecule(
        {
            "code": "M-002",
            "name": "ethanol",
            "smiles": "CCO",
            "parameters": {"family": "solvent", "viscosity": 1.2},
            "is_hidden": True,
        }
    )
    acetone = service.save_molecule(
        {
            "code": "M-003",
            "name": "acetone",
            "smiles": "CC(=O)C",
            "parameters": {"family": "ketone", "viscosity": 0.4},
        }
    )

    default_listing = service.list_molecules(sort_by="name", descending=False)
    assert [item["name"] for item in default_listing["items"]] == ["acetone", "glycol"]
    assert default_listing["total"] == 2

    include_hidden = service.list_molecules(include_hidden=True, sort_by="code", descending=False)
    assert [item["code"] for item in include_hidden["items"]] == ["M-001", "M-002", "M-003"]

    hidden_only = service.list_molecules(hidden_only=True)
    assert hidden_only["total"] == 1
    assert hidden_only["items"][0]["id"] == ethanol["id"]

    search_by_keyword = service.list_molecules(keyword="ketone", include_hidden=True)
    assert search_by_keyword["total"] == 1
    assert search_by_keyword["items"][0]["id"] == acetone["id"]

    search_by_code = service.list_molecules(keyword="M-001", include_hidden=True)
    assert search_by_code["items"][0]["id"] == glycol["id"]

    page_two = service.list_molecules(include_hidden=True, page=2, page_size=2, sort_by="code", descending=False)
    assert page_two["total"] == 3
    assert len(page_two["items"]) == 1

    detail = service.get_molecule_detail(glycol["id"])
    assert detail["parameters"]["family"] == "solvent"
    assert detail["canonical_smiles"]

    updated = service.save_molecule(
        {
            "code": "M-001",
            "name": "glycol-updated",
            "smiles": "OCCO",
            "parameters": {"family": "solvent", "viscosity": 13.0, "density": 1.11},
            "notes": "updated",
        },
        molecule_id=glycol["id"],
    )
    assert updated["name"] == "glycol-updated"
    updated_detail = service.get_molecule_detail(glycol["id"])
    assert updated_detail["parameters"]["density"] == "1.11"
    assert updated_detail["notes"] == "updated"

    shown = service.set_hidden_state(ethanol["id"], False)
    assert shown["is_hidden"] is False
    assert service.list_molecules()["total"] == 3

    assert service.delete_molecule(acetone["id"]) is True
    assert service.list_molecules(include_hidden=True)["total"] == 2
