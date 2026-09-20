"""Preprocessing utility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data.preprocessing import (
    assert_finite,
    drop_correlated_features,
    make_preprocessor,
    select_numeric_features,
)


def test_drop_correlated_features():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    df = pd.DataFrame({
        "a": a,
        "b": a * 2.0,                      # perfectly correlated with a -> dropped
        "c": rng.normal(size=500),         # independent -> kept
        "d": a + rng.normal(0, 0.01, 500), # ~1.0 correlation with a -> dropped
    })
    reduced, dropped = drop_correlated_features(df, threshold=0.95, return_dropped=True)
    assert list(reduced.columns) == ["a", "c"]
    assert dropped == ["b", "d"]


def test_drop_correlated_features_threshold_validation():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError):
        drop_correlated_features(df, threshold=0.0)
    with pytest.raises(ValueError):
        drop_correlated_features(df, threshold=1.5)


def test_make_preprocessor_handles_nan_and_scales():
    X = np.array([
        [1.0, 10.0],
        [np.nan, 20.0],
        [3.0, 30.0],
        [5.0, 40.0],
    ])
    pre = make_preprocessor(scale=True, impute=True)
    Xt = pre.fit_transform(X)
    assert np.isfinite(Xt).all()
    assert np.isnan(Xt).sum() == 0
    assert abs(Xt[:, 0].mean()) < 1e-9  # standardized
    assert 0.9 < Xt[:, 0].std() < 1.1


def test_make_preprocessor_identity():
    pre = make_preprocessor(scale=False, impute=False)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(pre.fit_transform(X), X)


def test_select_numeric_features():
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0],
        "num_str": ["1.5", "2.5", "3.5"],
        "text": ["a", "b", "c"],
        "all_nan_str": ["x", "y", "z"],
    })
    assert select_numeric_features(df) == ["num", "num_str"]


def test_assert_finite_raises():
    assert_finite(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        assert_finite(np.array([[np.nan, 2.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        assert_finite(np.array([[np.inf, 2.0]]))
