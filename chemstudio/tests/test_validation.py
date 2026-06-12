from __future__ import annotations

import pandas as pd
import pytest

from chemstudio.validation import (
    validate_feature_names,
    validate_molecule_name,
    validate_ratio,
    validate_smiles,
    validate_target_column,
)


def test_validate_smiles_returns_canonical_smiles():
    assert validate_smiles(" C(C)O ") == "CCO"


def test_validate_molecule_name_rejects_blank_and_long_values():
    assert validate_molecule_name(" ethanol ") == "ethanol"
    with pytest.raises(ValueError, match="required"):
        validate_molecule_name("   ")
    with pytest.raises(ValueError, match="500"):
        validate_molecule_name("x" * 501)


def test_validate_ratio_enforces_range():
    assert validate_ratio(33.3) == 33.3
    with pytest.raises(ValueError, match="负数"):
        validate_ratio(-1.0)
    with pytest.raises(ValueError, match="100"):
        validate_ratio(101.0)


def test_model_validators_check_target_and_feature_names():
    dataset = pd.DataFrame({"target": [1.0, None], "empty_target": [None, None]})
    validate_target_column(dataset, "target")
    with pytest.raises(ValueError, match="not found"):
        validate_target_column(dataset, "missing")
    with pytest.raises(ValueError, match="no usable"):
        validate_target_column(dataset, "empty_target")

    validate_feature_names(["a", "b"])
    validate_feature_names(["target"], list(dataset.columns))
    with pytest.raises(ValueError, match="No usable"):
        validate_feature_names([])
    with pytest.raises(ValueError, match="non-empty"):
        validate_feature_names(["a", " "])
    with pytest.raises(ValueError, match="not found"):
        validate_feature_names(["missing"], list(dataset.columns))
