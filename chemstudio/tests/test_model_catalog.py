from __future__ import annotations

import pytest

from chemstudio.ml import model_catalog


def test_xgboost_native_load_failure_is_reported_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xgboost_row = next(item for item in model_catalog.MODEL_CATALOG if item["key"] == "xgboost")
    monkeypatch.setitem(xgboost_row, "available", False)
    monkeypatch.setattr(model_catalog, "XGBRegressor", None)
    monkeypatch.setattr(model_catalog, "_XGBOOST_IMPORT_ERROR", OSError("missing libomp.dylib"))

    catalog_row = next(item for item in model_catalog.get_model_catalog() if item["key"] == "xgboost")
    assert catalog_row["available"] is False

    with pytest.raises(RuntimeError, match="brew install libomp"):
        model_catalog.create_regression_model("xgboost")
