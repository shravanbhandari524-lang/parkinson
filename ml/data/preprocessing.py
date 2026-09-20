"""Shared preprocessing utilities.

The scikit-learn pipelines built in :mod:`ml.models` embed the preprocessor
returned by :func:`make_preprocessor`, so imputation/scaling statistics are
fitted on training folds only — never on the full dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def select_numeric_features(df: pd.DataFrame) -> list[str]:
    """Return the names of columns that are numeric (or coercible to numeric)."""
    numeric: list[str] = []
    for col in df.columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().any():
            numeric.append(col)
    return numeric


def drop_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    return_dropped: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """Greedily drop features whose absolute Pearson correlation with an
    already-kept feature exceeds ``threshold`` (keeps the earlier column).

    Intended for linear baselines; tree ensembles are usually left untouched.
    """
    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be in (0, 1]")
    corr = df.corr(numeric_only=True).abs()
    kept: list[str] = []
    dropped: list[str] = []
    for col in df.columns:
        if any(float(corr.loc[col, kept_col]) > threshold for kept_col in kept):
            dropped.append(col)
        else:
            kept.append(col)
    reduced = df[kept].copy()
    return (reduced, dropped) if return_dropped else reduced


def make_preprocessor(scale: bool = True, impute: bool = True) -> Pipeline:
    """Build a preprocessor pipeline (median imputation + standardization).

    Fitted inside model Pipelines, so it is refit per CV fold — no leakage.
    """
    steps: list[tuple[str, object]] = []
    if impute:
        steps.append(("imputer", SimpleImputer(strategy="median", keep_empty_features=True)))
    if scale:
        steps.append(("scaler", StandardScaler()))
    if not steps:
        return Pipeline([("identity", "passthrough")])
    return Pipeline(steps)


def assert_finite(X: np.ndarray, name: str = "X") -> None:
    """Raise if a matrix still contains NaN/inf after preprocessing."""
    if not np.isfinite(X).all():
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        raise ValueError(f"{name} contains non-finite values ({n_nan} NaN, {n_inf} inf)")
