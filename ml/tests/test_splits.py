"""CV split and leakage-guard tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from ml.models.splits import (
    assert_no_group_leakage,
    default_strategy_for,
    group_overlap,
    iter_folds,
    make_cv,
)


def _toy(n_groups: int = 12, per_group: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.array([f"g{i}" for i in range(n_groups) for _ in range(per_group)])
    X = np.column_stack([rng.normal(size=len(groups)), rng.normal(size=len(groups))])
    # classes correlated with group parity so stratification has signal
    y = np.array([0 if (int(g[1:]) % 2 == 0) ^ (i % 3 == 0) else 1
                  for g, i in zip(groups, range(len(groups)))])
    return X, y, groups


def test_make_cv_strategies():
    X, y, groups = _toy()
    for strategy in ("stratified", "group", "stratified_group"):
        cv = make_cv(strategy, n_splits=4, seed=1)
        n = sum(1 for _ in iter_folds(cv, X, y, groups, check_group_leakage=(strategy != "stratified")))
        assert n == 4
    # LeaveOneGroupOut yields one fold per group by design
    cv = make_cv("logo")
    n = sum(1 for _ in iter_folds(cv, X, y, groups))
    assert n == 12


def test_group_split_has_no_leakage():
    X, y, groups = _toy()
    cv = make_cv("group", n_splits=4)
    for train_idx, test_idx in iter_folds(cv, X, y, groups):
        assert group_overlap(train_idx, test_idx, groups) == set()


def test_logo_every_group_held_out_once():
    X, y, groups = _toy(n_groups=10, per_group=4)
    cv = make_cv("logo")
    held_out: set = set()
    for train_idx, test_idx in iter_folds(cv, X, y, groups):
        out = set(groups[test_idx].tolist())
        assert len(out) == 1
        assert group_overlap(train_idx, test_idx, groups) == set()
        held_out |= out
    assert held_out == set(groups.tolist())


def test_leakage_guard_raises():
    groups = np.array(["a", "a", "b", "b"])
    # train {a, b} vs test {a, b} -> overlap
    with pytest.raises(AssertionError):
        assert_no_group_leakage(np.array([0, 2]), np.array([1, 3]), groups)


def test_check_flag_disables_guard():
    """Voice's record-level spec needs the guard off (subjects span folds)."""
    X, y, groups = _toy()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    folds = list(iter_folds(cv, X, y, groups, check_group_leakage=False))
    assert len(folds) == 3  # would raise if the guard were on


def test_default_strategy_mapping():
    assert default_strategy_for("voice") == "stratified"
    assert default_strategy_for("handwriting") == "group"
    assert default_strategy_for("gait") == "group"
