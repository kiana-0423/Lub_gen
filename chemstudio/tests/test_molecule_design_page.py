from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QSplitter

from chemstudio.ui.molecule_design_page import MoleculeDesignPage


class FakeDatabaseManager:
    pass


class FakeFeatureService:
    pass


class FakeModelService:
    def __init__(self) -> None:
        self.explain_calls = 0

    def get_training_dataset(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"id": 1, "name": "ethanol", "viscosity": 1.2, "feature_a": 0.1},
                {"id": 2, "name": "acetone", "viscosity": 0.3, "feature_a": 0.4},
            ]
        )

    def get_target_columns(self) -> list[str]:
        return ["viscosity"]

    def infer_problem_type(self, target_name: str) -> str:
        del target_name
        return "regression"

    def get_model_catalog(self, problem_type: str | None = None) -> list[dict[str, object]]:
        del problem_type
        return [
            {
                "key": "random_forest",
                "label": "RandomForestRegressor",
                "available": True,
            }
        ]

    def is_explainer_available(self) -> bool:
        return False

    def explain_model(self, artifact: dict[str, object]) -> object:
        del artifact
        self.explain_calls += 1
        return object()


class FakeVisualizationService:
    pass


class FakeMoleculeRepository:
    def list_feature_names(self) -> list[str]:
        return ["feature_a"]


@pytest.fixture()
def qt_app():
    app = QApplication.instance() or QApplication([])
    return app


def test_molecule_design_page_uses_three_column_workspace(qt_app) -> None:
    del qt_app
    page = MoleculeDesignPage(
        FakeDatabaseManager(),
        FakeFeatureService(),
        FakeModelService(),
        FakeVisualizationService(),
        FakeMoleculeRepository(),
    )

    workspace = page.findChild(QSplitter, "moleculeWorkspace")

    assert workspace is not None
    assert workspace.count() == 3
    assert workspace.orientation() == Qt.Orientation.Horizontal
    assert workspace.widget(0).maximumWidth() == 420
    assert workspace.widget(2).maximumWidth() == 360
    assert page.canvas.minimumHeight() >= 300
    assert isinstance(page.feature_selection_report_label, QPlainTextEdit)
    assert page.feature_text_edit.maximumHeight() == 180
    assert page.shap_explanation_box.parent() is workspace.widget(0)
    assert page.shap_explanation_box.isHidden()


def test_molecule_explanation_runs_without_qthread(monkeypatch: pytest.MonkeyPatch, qt_app) -> None:
    del qt_app
    model_service = FakeModelService()
    model_service.is_explainer_available = lambda: True  # type: ignore[method-assign]
    page = MoleculeDesignPage(
        FakeDatabaseManager(),
        FakeFeatureService(),
        model_service,
        FakeVisualizationService(),
        FakeMoleculeRepository(),
    )
    page.current_artifact = {"model": object(), "feature_names": ["feature_a"]}
    loaded: list[object] = []
    monkeypatch.setattr(page.shap_summary_widget, "load_explanation", lambda explanation: loaded.append(explanation))

    page._explain_model()

    assert model_service.explain_calls == 1
    assert page._explanation_thread is None
    assert loaded
    assert not page.shap_explanation_box.isHidden()
