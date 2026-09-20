"""Metric-suite tests with analytically known outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from ml.metrics import aggregate_fold_metrics, compute_metrics


def test_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    m = compute_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0 and m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_all_positive_predictions():
    y_true = [0, 0, 1, 1]
    y_prob = [0.9, 0.8, 0.7, 0.6]  # all above threshold -> predict 1
    m = compute_metrics(y_true, y_prob)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 0.0
    assert m["precision"] == 0.5
    assert m["confusion_matrix"] == {"tn": 0, "fp": 2, "fn": 0, "tp": 2}


def test_threshold_moves_predictions():
    y_true = [0, 1]
    m_low = compute_metrics(y_true, [0.4, 0.6], threshold=0.5)
    m_high = compute_metrics(y_true, [0.4, 0.6], threshold=0.7)
    assert m_low["accuracy"] == 1.0
    assert m_high["specificity"] == 1.0 and m_high["sensitivity"] == 0.0


def test_roc_auc_single_class_is_none():
    m = compute_metrics([1, 1, 1], [0.9, 0.8, 0.7])
    assert m["roc_auc"] is None


def test_confusion_matrix_values():
    y_true = [0, 0, 0, 1, 1]
    y_prob = [0.9, 0.8, 0.2, 0.4, 0.6]
    m = compute_metrics(y_true, y_prob)
    # pred: 1,1,0,0,1 -> tn=1 fp=2 fn=1 tp=1
    assert m["confusion_matrix"] == {"tn": 1, "fp": 2, "fn": 1, "tp": 1}
    assert m["sensitivity"] == 0.5 and m["specificity"] == 1 / 3


def test_aggregate_fold_metrics():
    folds = [
        {"fold": 1, "accuracy": 0.8, "roc_auc": 0.9, "confusion_matrix": {"tn": 1}},
        {"fold": 2, "accuracy": 0.6, "roc_auc": 0.7, "confusion_matrix": {"tn": 2}},
    ]
    agg = aggregate_fold_metrics(folds)
    assert agg["accuracy_mean"] == pytest.approx(0.7)
    assert agg["roc_auc_std"] == pytest.approx(0.14142135623730953)  # sqrt(0.02)
    assert agg["confusion_matrix"] is None
