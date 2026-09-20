"""Baseline fusion over per-modality out-of-fold probabilities.

The three source datasets are disjoint cohorts (UCI voice subjects, HandPD
patients, PhysioNet subjects) — no patient exists in more than one modality.
Fusion therefore happens at *score level*: every row of the meta-dataset
carries

* the out-of-fold probability of its own modality's primary model, and
* missing-value indicators for the modalities that were not measured,

so the combiners learn a decision rule that also works when only one
modality is available (the deployment case for a new patient measured on a
subset of modalities).  All inputs are out-of-fold probabilities, and the
fusion CV itself is grouped by patient/subject, so no information from a
validation subject ever touches a training fold.

Two baselines are implemented per the project spec:

1. weighted soft voting (weights fitted per fold by a simplex grid search
   maximising train-fold ROC-AUC; an equal-weights variant is reported too)
2. logistic-regression stacking over the three probabilities + indicators
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import Config, get_config
from ml.metrics import compute_metrics, save_metrics_report
from ml.metrics.metrics import aggregate_fold_metrics

MODALITIES = ("voice", "handwriting", "gait")
NEUTRAL_PROB = 0.5  # fill for missing modality probabilities


# ---------------------------------------------------------------------------
# meta-dataset
# ---------------------------------------------------------------------------
def build_meta_dataset(cfg: Config | None = None) -> pd.DataFrame:
    """Stack the three modality OOF tables into one score-level frame.

    Columns: ``y``, ``group`` (modality-prefixed), ``modality``,
    ``<modality>_prob`` for every modality (NaN when not measured) and
    ``<modality>_missing`` indicators.
    """
    cfg = cfg or get_config()
    parts: list[pd.DataFrame] = []
    for modality in MODALITIES:
        metrics_path = cfg.metrics_dir / f"{modality}_metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"missing {metrics_path} — train the {modality} modality first "
                f"(run ml/train_all.py or ml.models.{modality}.train_{modality}())"
            )
        import json

        with open(metrics_path, encoding="utf-8") as fh:
            primary_model = json.load(fh)["primary_model"]
        oof = pd.read_csv(cfg.metrics_dir / f"oof_{modality}.csv", index_col="row_id")
        if primary_model not in oof.columns:
            raise KeyError(f"OOF table for {modality} lacks primary column {primary_model!r}")

        part = pd.DataFrame(index=oof.index)
        part["y"] = oof["y"].astype(int)
        part["group"] = f"{modality}::" + oof["group"].astype(str)
        part["modality"] = modality
        for m in MODALITIES:
            part[f"{m}_prob"] = oof[primary_model].astype(float) if m == modality else np.nan
            part[f"{m}_missing"] = 1.0 if m != modality else 0.0
        parts.append(part)

    meta = pd.concat(parts, ignore_index=True)
    prob_cols = [f"{m}_prob" for m in MODALITIES]
    if meta[prob_cols].notna().sum(axis=1).lt(1).any():
        raise RuntimeError("meta rows without any modality probability")
    return meta


def meta_feature_matrix(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, feature_names)`` with NaN probabilities neutral-filled."""
    prob_cols = [f"{m}_prob" for m in MODALITIES]
    indicator_cols = [f"{m}_missing" for m in MODALITIES]
    X = meta[prob_cols + indicator_cols].to_numpy(dtype=float, copy=True)
    X[:, : len(prob_cols)] = np.nan_to_num(X[:, : len(prob_cols)], nan=NEUTRAL_PROB)
    return X, None, prob_cols + indicator_cols


# ---------------------------------------------------------------------------
# combiner 1: weighted soft voting
# ---------------------------------------------------------------------------
def _score_matrix(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted vote over *available* modalities, renormalised per row.

    ``probs``: (n, 3) with NaN for unavailable modalities.
    ``weights``: (k, 3) candidate weight vectors -> (n, k) scores.
    """
    avail = ~np.isnan(probs)
    filled = np.nan_to_num(probs, nan=0.0)
    num = filled @ weights.T
    den = avail.astype(float) @ weights.T
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return scores


def simplex_grid(steps: tuple[float, ...] = (0.0, 0.33, 0.5, 1.0, 2.0, 3.0)) -> np.ndarray:
    """All non-zero weight vectors over the modalities from ``steps``."""
    combos = np.array([w for w in product(steps, repeat=len(MODALITIES)) if sum(w) > 0])
    return combos


@dataclass
class WeightedSoftVoter:
    """Soft-voting combiner with per-modality weights fitted on train folds."""

    weights: np.ndarray  # (3,) aligned with MODALITIES
    grid_steps: tuple[float, ...] = (0.0, 0.33, 0.5, 1.0, 2.0, 3.0)

    @classmethod
    def equal(cls) -> "WeightedSoftVoter":
        return cls(weights=np.ones(len(MODALITIES)))

    @classmethod
    def fit(cls, probs: np.ndarray, y: np.ndarray,
            grid_steps: tuple[float, ...] = (0.0, 0.33, 0.5, 1.0, 2.0, 3.0)) -> "WeightedSoftVoter":
        """Pick the weight vector maximising ROC-AUC on the training rows.

        Weight vectors that give zero total weight to some row's available
        modalities (they would score that row NaN) are rejected outright so
        the fitted combiner is total over every availability pattern.
        """
        grid = simplex_grid(grid_steps)
        scores = _score_matrix(probs, grid)
        aucs = np.full(grid.shape[0], -1.0)
        for j in range(scores.shape[1]):
            if np.isnan(scores[:, j]).any():
                continue  # vector cannot score every availability pattern
            if len(set(y.tolist())) > 1:
                aucs[j] = roc_auc_score(y, scores[:, j])
            else:
                aucs[j] = 0.5
        best = int(np.argmax(aucs))
        if aucs[best] < 0:
            raise RuntimeError("no admissible weight vector in the simplex grid")
        return cls(weights=grid[best], grid_steps=grid_steps)

    def decision_function(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        scores = _score_matrix(probs, self.weights[None, :])[:, 0]
        if np.isnan(scores).any():
            # deployment safety net: rows whose available modalities all carry
            # zero fitted weight fall back to an equal-weight vote
            fallback = _score_matrix(probs, np.ones((1, len(MODALITIES))))[:, 0]
            scores = np.where(np.isnan(scores), fallback, scores)
        return scores

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        """Return ``[[1-p, p], ...]`` (scores are already in [0, 1])."""
        p = np.clip(self.decision_function(probs), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def to_dict(self) -> dict:
        return {m: float(w) for m, w in zip(MODALITIES, self.weights)}


# ---------------------------------------------------------------------------
# combiner 2: logistic-regression stacking
# ---------------------------------------------------------------------------
def make_stacking_lr(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                                     random_state=seed)),
    ])


# ---------------------------------------------------------------------------
# evaluation driver
# ---------------------------------------------------------------------------
def evaluate_fusion(cfg: Config | None = None,
                    n_splits: int = 5,
                    seed: int = 42,
                    include_tabnet: bool = False) -> dict:
    """Grouped CV of the baseline fusion combiners over the meta-dataset.

    Rows from the same patient/subject never straddle train and validation
    folds; the OOF probabilities feeding the combiners were themselves
    produced without leakage by the per-modality stage.
    """
    cfg = cfg or get_config()
    meta = build_meta_dataset(cfg)
    X, _, feature_names = meta_feature_matrix(meta)
    y = meta["y"].to_numpy(dtype=int)
    groups = meta["group"].to_numpy()
    probs_raw = meta[[f"{m}_prob" for m in MODALITIES]].to_numpy(dtype=float)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results: dict[str, dict] = {}

    def _evaluate(name: str, fold_rows: list[dict], weights: dict | None = None) -> None:
        results[name] = {
            "aggregate": aggregate_fold_metrics(fold_rows),
            "fold_metrics": fold_rows,
            **({"weights": weights} if weights is not None else {}),
        }

    weighted_folds: list[dict] = []
    lr_folds: list[dict] = []
    fitted_weights: dict | None = None
    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups=groups), start=1):
        voter = WeightedSoftVoter.fit(probs_raw[tr], y[tr])
        fitted_weights = voter.to_dict()
        prob_te = voter.predict_proba(probs_raw[te])[:, 1]
        weighted_folds.append({"fold": fold_idx, **compute_metrics(y[te], prob_te)})

        lr = make_stacking_lr(seed)
        lr.fit(X[tr], y[tr])
        prob_te_lr = lr.predict_proba(X[te])[:, 1]
        lr_folds.append({"fold": fold_idx, **compute_metrics(y[te], prob_te_lr)})
    _evaluate("weighted_soft_voting", weighted_folds, weights=fitted_weights)
    _evaluate("weighted_soft_voting_equal",
              [{"fold": 0,
                **compute_metrics(y, WeightedSoftVoter.equal().predict_proba(probs_raw)[:, 1])}],
              weights=WeightedSoftVoter.equal().to_dict())
    _evaluate("logistic_regression_stacking", lr_folds)

    # refit combiners on the full meta-dataset and persist
    artifacts_dir = cfg.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    import joblib

    final_voter = WeightedSoftVoter.fit(probs_raw, y)
    final_lr = make_stacking_lr(seed).fit(X, y)
    joblib.dump(final_voter, artifacts_dir / "fusion_weighted_soft_voting.joblib")
    joblib.dump(final_lr, artifacts_dir / "fusion_logistic_regression_stacking.joblib")

    tabnet_block: dict = {}
    if include_tabnet:
        from ml.models.tabnet_model import evaluate_tabnet_fusion

        tabnet_block = evaluate_tabnet_fusion(cfg, meta=meta, seed=seed)

    meta_path = cfg.metrics_dir / "fusion_meta_dataset.csv"
    meta.to_csv(meta_path, index_label="row_id")

    report = {
        "modality": "fusion",
        "design": (
            "score-level late fusion over per-modality out-of-fold primary-model "
            "probabilities; disjoint cohorts so each row carries one modality "
            "score + missing-modality indicators; grouped CV over patients"
        ),
        "n_samples": int(len(y)),
        "n_groups": int(pd.unique(groups).size),
        "class_counts": {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "feature_names": feature_names,
        "n_folds": n_splits,
        "baseline_fusion": results,
        "proposed_model": "tabnet_fusion",
        "tabnet": {"report": tabnet_block["report"]} if tabnet_block else None,
        "meta_dataset": str(meta_path),
        "artifacts": {
            "weighted_soft_voting": str(artifacts_dir / "fusion_weighted_soft_voting.joblib"),
            "logistic_regression_stacking": str(artifacts_dir / "fusion_logistic_regression_stacking.joblib"),
        },
        "novelty_note": (
            "TabNet is the proposed multimodal fusion model; it was not identified "
            "in the supplied literature matrix (no claim of global novelty)."
        ),
    }
    save_metrics_report("fusion_metrics", report, cfg.metrics_dir)
    return report


__all__ = [
    "MODALITIES",
    "NEUTRAL_PROB",
    "WeightedSoftVoter",
    "build_meta_dataset",
    "meta_feature_matrix",
    "make_stacking_lr",
    "evaluate_fusion",
    "simplex_grid",
]
