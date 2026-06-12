from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from chemstudio.ml.explainer import SHAPExplanation
from chemstudio.ui.widgets.shap_canvas import SHAPSummaryWidget


@pytest.fixture()
def qt_app():
    app = QApplication.instance() or QApplication([])
    return app


def test_shap_summary_widget_enables_zoom_for_existing_plot(tmp_path, qt_app) -> None:
    del qt_app
    image_path = tmp_path / "summary.png"
    pixmap = QPixmap(640, 360)
    pixmap.fill(Qt.GlobalColor.white)
    assert pixmap.save(str(image_path))

    widget = SHAPSummaryWidget()
    explanation = SHAPExplanation(
        shap_values=np.array([[0.1, 0.2]]),
        feature_names=["feature_a", "feature_b"],
        base_value=0.0,
        global_importance={"feature_b": 0.2, "feature_a": 0.1},
        summary_plot_path=str(image_path),
        problem_type="regression",
    )

    widget.load_explanation(explanation)

    assert widget.open_summary_button.isEnabled()
    assert widget.summary_label.pixmap() is not None
    assert widget.importance_table.rowCount() == 2
