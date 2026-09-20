"""Fusion-layer unit tests (synthetic meta-frames — never used for training)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.models.fusion import (
    MODALITIES,
    NEUTRAL_PROB,
    WeightedSoftVoter,
    meta_feature_matrix,
    simplex_grid,
    _score_matrix,
)


def _synthetic_meta(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        modality = MODALITIES[i % len(MODALITIES)]
        y = int(rng.random() < 0.5)
        row = {"y": y, "group": f"{modality}::p{i}", "modality": modality}
        for m in MODALITIES:
            if m == modality:
                row[f"{m}_prob"] = float(np.clip(y + rng.normal(0, 0.15), 0, 1))
                row[f"{m}_missing"] = 0.0
            else:
                row[f"{m}_prob"] = np.nan
                row[f"{m}_missing"] = 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_simplex_grid_nonzero():
    grid = simplex_grid((0.0, 1.0))
    assert len(grid) == 7  # 2^3 - 1
    assert (grid.sum(axis=1) > 0).all()


def test_score_matrix_renormalises_over_available():
    probs = np.array([[0.8, np.nan, np.nan], [np.nan, np.nan, 0.6]])
    weights = np.array([[1.0, 3.0, 2.0]])
    scores = _score_matrix(probs, weights)
    assert np.isnan(scores).sum() == 0
    assert scores[0, 0] == pytest.approx(0.8)
    assert scores[1, 0] == pytest.approx(0.6)


def test_score_matrix_nan_when_zero_weight_on_available():
    probs = np.array([[np.nan, np.nan, 0.6]])
    scores = _score_matrix(probs, np.array([[1.0, 1.0, 0.0]]))
    assert np.isnan(scores[0, 0])


def test_voter_fit_rejects_partial_vectors():
    probs = np.array([[0.8, np.nan, np.nan], [np.nan, 0.7, np.nan]])
    y = np.array([1, 0])
    voter = WeightedSoftVoter.fit(probs, y, grid_steps=(0.0, 1.0, 2.0))
    scores = voter.decision_function(probs)
    assert np.isnan(scores).sum() == 0


def test_voter_equal_weights_recovers_mean():
    probs = np.array([[0.8, 0.4, np.nan]])
    voter = WeightedSoftVoter.equal()
    assert voter.decision_function(probs)[0] == pytest.approx(0.6)


def test_voter_predict_proba_shape_and_range():
    probs = np.array([[0.8, np.nan, np.nan], [np.nan, np.nan, 0.6]])
    voter = WeightedSoftVoter(np.array([1.0, 1.0, 1.0]))
    proba = voter.predict_proba(probs)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_meta_feature_matrix_fill():
    meta = _synthetic_meta(n=6)
    X, _, names = meta_feature_matrix(meta)
    assert X.shape == (6, 6)
    assert not np.isnan(X).any()  # probabilities neutral-filled
    assert names == [f"{m}_prob" for m in MODALITIES] + [f"{m}_missing" for m in MODALITIES]
    prob_block = X[:, :3]
    # each row has exactly one real probability, the rest neutral
    for row in prob_block:
        real = row[row != NEUTRAL_PROB]
        assert len(real) == 1
        assert 0.0 <= real[0] <= 1.0
