from __future__ import annotations

import pandas as pd
import pytest

from chemstudio.ml.feature_selection import FeatureSelector


def test_variance_filter_removes_constant_features():
    x_frame = pd.DataFrame(
        {
            "mol_wt": [1.0, 1.0, 1.0, 1.0],
            "constant_descriptor": [5.0, 5.0, 5.0, 5.0],
            "moving_descriptor": [0.0, 1.0, 2.0, 3.0],
        }
    )
    y_values = pd.Series([0.0, 1.0, 2.0, 3.0])

    selected_features, report = FeatureSelector(strategy="variance").select(x_frame, y_values, "regression")

    assert "constant_descriptor" not in selected_features
    assert "moving_descriptor" in selected_features
    assert "mol_wt" in selected_features
    assert report.stages[0].name == "variance"


def test_correlation_filter_removes_highly_correlated():
    x_frame = pd.DataFrame(
        {
            "first_signal": [1.0, 2.0, 3.0, 4.0, 5.0],
            "duplicate_signal": [2.0, 4.0, 6.0, 8.0, 10.0],
            "independent_signal": [5.0, 3.0, 4.0, 1.0, 2.0],
        }
    )
    y_values = pd.Series([1.0, 2.0, 1.5, 3.0, 2.5])

    selected_features, report = FeatureSelector(strategy="correlation").select(x_frame, y_values, "regression")

    assert len({"first_signal", "duplicate_signal"} & set(selected_features)) == 1
    assert "independent_signal" in selected_features
    assert any(stage.name == "correlation" for stage in report.stages)


def test_univariate_filter_respects_max_features():
    x_frame = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            "feature_c": [1.0, 1.1, 1.2, 4.0, 4.1, 4.2],
            "feature_d": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        }
    )
    y_values = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    selected_features, report = FeatureSelector(strategy="univariate", max_features=2).select(
        x_frame,
        y_values,
        "regression",
    )

    assert len(selected_features) <= 2
    assert report.final_feature_count == len(selected_features)


def test_model_based_filter_with_classification():
    x_frame = pd.DataFrame(
        {
            "class_signal": [0.0, 0.1, 1.0, 1.1, 0.2, 1.2, 0.3, 1.3],
            "secondary_signal": [0.2, 0.1, 1.2, 1.0, 0.4, 1.4, 0.3, 1.1],
            "noise": [3.0, 1.0, 2.0, 5.0, 4.0, 0.0, 6.0, 7.0],
        }
    )
    y_values = pd.Series([0, 0, 1, 1, 0, 1, 0, 1])

    selected_features, report = FeatureSelector(strategy="model_based", max_features=2).select(
        x_frame,
        y_values,
        "classification",
    )

    assert selected_features
    assert report.problem_type == "classification"
    assert any(stage.name == "model_based" for stage in report.stages)


def test_strategy_none_returns_all_features():
    x_frame = pd.DataFrame({"feature_a": [1.0, 2.0], "feature_b": [3.0, 4.0]})
    y_values = pd.Series([0.0, 1.0])

    selected_features, report = FeatureSelector(strategy="none").select(x_frame, y_values, "regression")

    assert selected_features == ["feature_a", "feature_b"]
    assert report.stages == []


def test_protected_features_never_removed():
    x_frame = pd.DataFrame(
        {
            "mol_wt": [1.0, 1.0, 1.0, 1.0],
            "mol_logp": [2.0, 2.0, 2.0, 2.0],
            "tpsa": [3.0, 3.0, 3.0, 3.0],
            "constant_descriptor": [4.0, 4.0, 4.0, 4.0],
        }
    )
    y_values = pd.Series([0.0, 1.0, 2.0, 3.0])

    selected_features, _ = FeatureSelector(strategy="full", max_features=1).select(x_frame, y_values, "regression")

    assert {"mol_wt", "mol_logp", "tpsa"}.issubset(set(selected_features))
    assert "constant_descriptor" not in selected_features


def test_empty_frame_raises_error():
    with pytest.raises(ValueError, match="at least one feature"):
        FeatureSelector(strategy="full").select(pd.DataFrame(), pd.Series(dtype=float), "regression")


def test_report_stages_record_each_step():
    x_frame = pd.DataFrame(
        {
            "mol_wt": [18.0, 46.0, 41.0, 90.0, 88.0, 102.0, 120.0, 74.0],
            "mol_logp": [-1.3, -0.3, -0.4, 0.2, -0.2, 0.1, 0.5, -0.1],
            "tpsa": [31.5, 20.2, 23.8, 35.5, 35.4, 35.6, 40.0, 28.0],
            "duplicate_mol_wt": [18.0, 46.0, 41.0, 90.0, 88.0, 102.0, 120.0, 74.0],
            "constant_descriptor": [1.0] * 8,
            "sparse_descriptor": [1.0, None, None, None, 2.0, None, None, None],
            "useful_descriptor": [0.1, 0.3, 0.2, 0.7, 0.8, 0.9, 1.0, 0.6],
        }
    )
    y_values = pd.Series([1.0, 1.2, 0.9, 1.6, 1.7, 2.0, 2.4, 1.5])

    selected_features, report = FeatureSelector(strategy="full", max_features=3).select(
        x_frame,
        y_values,
        "regression",
    )
    report_payload = report.to_dict()

    assert {"mol_wt", "mol_logp", "tpsa"}.issubset(set(selected_features))
    assert "constant_descriptor" not in selected_features
    assert "sparse_descriptor" not in selected_features
    assert report_payload["initial_feature_count"] == 7
    assert report_payload["final_feature_count"] == len(selected_features)
    assert [stage["name"] for stage in report_payload["stages"]] == [
        "variance",
        "missing",
        "correlation",
        "univariate",
        "model_based",
    ]


def test_variance_and_missing_filters_use_cache_on_repeated_selection():
    x_frame = pd.DataFrame(
        {
            "mol_wt": [10.0, 11.0, 12.0, 13.0],
            "constant_descriptor": [1.0, 1.0, 1.0, 1.0],
            "sparse_descriptor": [1.0, None, None, None],
            "signal": [0.0, 1.0, 2.0, 3.0],
        }
    )
    y_values = pd.Series([0.0, 1.0, 2.0, 3.0])
    selector = FeatureSelector(strategy="correlation")

    _, first_report = selector.select(x_frame, y_values, "regression")
    _, second_report = selector.select(x_frame, y_values, "regression")

    assert not first_report.stages[0].from_cache
    assert not first_report.stages[1].from_cache
    assert second_report.stages[0].from_cache
    assert second_report.stages[1].from_cache
    assert second_report.to_dict()["stages"][0]["from_cache"] is True

    selector.clear_cache()
    _, third_report = selector.select(x_frame, y_values, "regression")
    assert not third_report.stages[0].from_cache
