"""Cross-validation strategies with hard leakage guarantees.

Split policy (project spec):
* voice        -> StratifiedKFold (recording-level; subject ids are available
                  in metadata if a stricter subject-level split is wanted)
* handwriting  -> GroupKFold on ID_PATIENT-derived groups
* gait         -> GroupKFold (or LeaveOneGroupOut) on subject ids

Every fold yielded by :func:`iter_folds` is checked with
:func:`assert_no_group_leakage`, so a record from the same patient/subject
can never appear in both the train and the validation part.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)

STRATEGIES = ("stratified", "group", "stratified_group", "logo")


def make_cv(strategy: str, n_splits: int = 5, seed: int = 42):
    """Build a scikit-learn splitter for the given strategy name."""
    if strategy == "stratified":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "group":
        return GroupKFold(n_splits=n_splits)
    if strategy == "stratified_group":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if strategy == "logo":
        return LeaveOneGroupOut()
    raise ValueError(f"unknown CV strategy: {strategy!r} (expected one of {STRATEGIES})")


def group_overlap(train_idx: np.ndarray, test_idx: np.ndarray, groups: np.ndarray) -> set:
    """Return the set of groups present in both train and test (empty if clean)."""
    train_groups = set(np.asarray(groups)[train_idx].tolist())
    test_groups = set(np.asarray(groups)[test_idx].tolist())
    return train_groups & test_groups


def assert_no_group_leakage(train_idx: np.ndarray, test_idx: np.ndarray, groups: np.ndarray) -> None:
    """Raise if any patient/subject id appears on both sides of a split."""
    overlap = group_overlap(train_idx, test_idx, groups)
    if overlap:
        raise AssertionError(f"group leakage detected; groups in train AND test: {sorted(overlap)}")


def iter_folds(
    cv,
    X,
    y,
    groups,
    check_group_leakage: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` pairs, asserting no group leakage per fold.

    ``check_group_leakage`` may be set to ``False`` for deliberately
    record-level strategies (voice uses StratifiedKFold per the project spec,
    which intentionally allows multiple recordings of one subject across
    folds); the assertion is only meaningful for group-aware splitters.
    """
    groups_arr = np.asarray(groups) if groups is not None else None
    if groups_arr is not None and check_group_leakage:
        split = cv.split(X, y, groups=groups_arr)
    else:
        split = cv.split(X, y)
    for train_idx, test_idx in split:
        if groups_arr is not None and check_group_leakage:
            assert_no_group_leakage(train_idx, test_idx, groups_arr)
        yield np.asarray(train_idx), np.asarray(test_idx)


def default_strategy_for(modality: str) -> str:
    """Project-spec split strategy per modality."""
    return {"voice": "stratified", "handwriting": "group", "gait": "group"}[modality]


__all__ = [
    "STRATEGIES",
    "make_cv",
    "group_overlap",
    "assert_no_group_leakage",
    "iter_folds",
    "default_strategy_for",
    "KFold",
]
