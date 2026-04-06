from __future__ import annotations

import pytest

from chemstudio.services.visualization_service import VisualizationService


@pytest.mark.skipif(
    VisualizationService.generate_3d_molblock.__globals__["Chem"] is None,
    reason="RDKit not installed",
)
def test_generate_3d_molblock_from_valid_smiles():
    service = VisualizationService()

    molblock, error_message = service.generate_3d_molblock("CCO")

    assert error_message is None
    assert molblock is not None
    assert "M  END" in molblock


@pytest.mark.skipif(
    VisualizationService.generate_3d_molblock.__globals__["Chem"] is None,
    reason="RDKit not installed",
)
def test_generate_3d_molblock_returns_friendly_error_for_invalid_smiles():
    service = VisualizationService()

    molblock, error_message = service.generate_3d_molblock("not-a-smiles")

    assert molblock is None
    assert error_message == "无法解析当前 SMILES，不能生成 3D 结构。"


def test_build_molecule_viewer_html_contains_safe_payload():
    service = VisualizationService()

    html = service.build_molecule_viewer_html(
        molblock="mock molblock",
        molecule_name="<demo>",
        smiles="CCO",
    )

    assert "&lt;demo&gt;" in html
    assert "mock molblock" in html
    assert "renderBuiltinViewer" in html
    assert "resetBuiltinView" in html
    assert "3Dmol" not in html
