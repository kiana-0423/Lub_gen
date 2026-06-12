from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication

from chemstudio.ui import data_page as data_page_module
from chemstudio.ui.data_page import DataPage


class FakeDatabaseManager:
    pass


class FakeMoleculeRepository:
    def __init__(self, dataframe: pd.DataFrame | None = None, descriptor_count: int = 0) -> None:
        self.dataframe = dataframe if dataframe is not None else pd.DataFrame(columns=["id", "name", "smiles"])
        self.descriptor_count = descriptor_count
        self.last_search_text = ""
        self.deleted_ids: list[int] = []

    def get_wide_dataset(self, search_text: str = "", *, include_mordred: bool = False) -> pd.DataFrame:
        self.last_search_text = search_text
        del include_mordred
        return self.dataframe.copy()

    def count_descriptor_rows(self) -> int:
        return self.descriptor_count

    def delete_molecule(self, molecule_id: int) -> bool:
        self.deleted_ids.append(molecule_id)
        return True


class FakeImportService:
    pass


class FakeVisualizationService:
    def build_molecule_viewer_html(self, **kwargs) -> str:
        del kwargs
        return "<html></html>"

    def generate_3d_molblock(self, smiles: str) -> tuple[str | None, str | None]:
        del smiles
        return None, "not rendered in tests"


@pytest.fixture()
def qt_app():
    app = QApplication.instance() or QApplication([])
    return app


def make_page(monkeypatch, dataframe: pd.DataFrame | None = None, descriptor_count: int = 0) -> DataPage:
    monkeypatch.setattr(data_page_module, "QWebEngineView", None)
    db_manager = FakeDatabaseManager()
    molecule_repository = FakeMoleculeRepository(dataframe=dataframe, descriptor_count=descriptor_count)
    page = DataPage(db_manager, FakeImportService(), FakeVisualizationService(), molecule_repository)
    page._fake_molecule_repository = molecule_repository
    return page


def test_export_button_disabled_when_mordred_unavailable(monkeypatch, qt_app):
    del qt_app
    monkeypatch.setattr(data_page_module, "is_mordred_available", lambda: False)

    page = make_page(monkeypatch)

    assert not page.export_features_button.isEnabled()
    assert page.export_features_button.toolTip() == "Mordred 未安装，无法导出描述符特征"


def test_export_button_enabled_when_mordred_available(monkeypatch, qt_app):
    del qt_app
    monkeypatch.setattr(data_page_module, "is_mordred_available", lambda: True)

    page = make_page(monkeypatch)

    assert page.export_features_button.isEnabled()
    assert page.export_features_button.toolTip() == "导出当前筛选后的 Mordred 特征宽表"


def test_export_csv_writes_utf8_bom(monkeypatch, qt_app, tmp_path):
    del qt_app
    monkeypatch.setattr(data_page_module, "is_mordred_available", lambda: True)
    dataframe = pd.DataFrame([{"id": 1, "name": "水", "smiles": "O", "ABC": 1.5}])
    page = make_page(monkeypatch, dataframe=dataframe, descriptor_count=1)
    page.search_input.setText("水")
    destination = tmp_path / "features.csv"

    row_count, column_count = page._write_features_csv(destination, page.search_input.text())

    assert (row_count, column_count) == (1, 4)
    assert destination.read_bytes().startswith(b"\xef\xbb\xbf")
    assert page._fake_molecule_repository.last_search_text == "水"


def test_export_csv_shows_warning_when_no_descriptors(monkeypatch, qt_app, tmp_path):
    del qt_app
    monkeypatch.setattr(data_page_module, "is_mordred_available", lambda: True)
    monkeypatch.setattr(
        data_page_module.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(tmp_path / "features.csv"), "CSV Files (*.csv)"),
    )
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        data_page_module.QMessageBox,
        "information",
        lambda parent, title, message: messages.append((title, message)),
    )
    page = make_page(
        monkeypatch,
        dataframe=pd.DataFrame([{"id": 1, "name": "water", "smiles": "O"}]),
        descriptor_count=0,
    )

    page._export_features_csv()

    assert messages == [("没有可导出的特征", "当前没有可导出的特征数据，请先导入分子并计算描述符。")]
    assert not (tmp_path / "features.csv").exists()


def test_default_export_path_uses_desktop_and_date(monkeypatch, qt_app, tmp_path):
    del qt_app
    monkeypatch.setattr(data_page_module, "is_mordred_available", lambda: True)
    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    monkeypatch.setattr(data_page_module.Path, "home", classmethod(lambda cls: tmp_path))
    page = make_page(monkeypatch)

    default_path = page._default_feature_export_path()

    assert default_path.parent == desktop
    assert default_path.name.startswith("mordred_features_")
    assert default_path.suffix == ".csv"
