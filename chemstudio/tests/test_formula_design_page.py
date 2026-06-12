from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtTest")

from PySide6.QtTest import QSignalSpy

from chemstudio.ui.formula_design_page import FormulaTrainingWorker


class FakeFormulaTrainingService:
    def train_formulation_model(self, **parameters):
        return {
            "model_name": "LinearRegression",
            "target_name": parameters["target_name"],
            "feature_names": ["molecule_1"],
            "metrics": {"r2": 1.0, "mae": 0.0, "rmse": 0.0},
            "sample_count": 4,
        }


def test_formula_training_worker_emits_finished_signal():
    worker = FormulaTrainingWorker(
        FakeFormulaTrainingService(),
        {"target_name": "conductivity", "model_key": "linear_regression"},
    )
    finished_spy = QSignalSpy(worker.finished)
    failed_spy = QSignalSpy(worker.failed)

    worker.run()

    assert finished_spy.count() == 1
    assert failed_spy.count() == 0
    assert finished_spy.at(0)[0]["target_name"] == "conductivity"
