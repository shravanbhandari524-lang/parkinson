"""Shared cross-validated training engine and model factories.

Every model is a scikit-learn Pipeline (preprocessor -> estimator) so that
imputation/scaling is refit inside each training fold — statistics from
validation data never touch the fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ml.config import Config
from ml.data.preprocessing import make_preprocessor
from ml.metrics import aggregate_fold_metrics, save_metrics_report
from ml.metrics import compute_metrics as _compute_metrics
from ml.models.splits import default_strategy_for, iter_folds, make_cv


# ---------------------------------------------------------------------------
# model factories
# ---------------------------------------------------------------------------
def make_xgboost(scale_pos_weight: float, seed: int) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor(impute=True, scale=False)),
        ("model", XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )),
    ])


def make_random_forest(scale_pos_weight: float, seed: int) -> Pipeline:
    """Random Forest; ``scale_pos_weight`` unused (class_weight='balanced')."""
    return Pipeline([
        ("preprocessor", make_preprocessor(impute=True, scale=False)),
        ("model", RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )),
    ])


def make_logistic_regression(scale_pos_weight: float, seed: int) -> Pipeline:
    """Logistic Regression; ``scale_pos_weight`` unused (class_weight='balanced')."""
    return Pipeline([
        ("preprocessor", make_preprocessor(impute=True, scale=True)),
        ("model", LogisticRegression(
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def make_svm(scale_pos_weight: float, seed: int) -> Pipeline:
    """RBF SVM; ``scale_pos_weight`` unused (class_weight='balanced')."""
    return Pipeline([
        ("preprocessor", make_preprocessor(impute=True, scale=True)),
        ("model", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


# ---------------------------------------------------------------------------
# specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str                       # "primary" | "baseline"
    factory: Callable[[float, int], Pipeline]  # (scale_pos_weight, seed) -> Pipeline


def build_model_specs(modality: str, seed: int) -> list[ModelSpec]:
    """Model roster per modality (project spec)."""
    if modality == "voice":
        return [
            ModelSpec("xgboost", "primary", make_xgboost),
            ModelSpec("logistic_regression", "baseline", make_logistic_regression),
            ModelSpec("random_forest", "baseline", make_random_forest),
        ]
    if modality == "handwriting":
        return [
            ModelSpec("random_forest", "primary", make_random_forest),
            ModelSpec("svm", "baseline", make_svm),
            ModelSpec("logistic_regression", "baseline", make_logistic_regression),
        ]
    if modality == "gait":
        return [
            ModelSpec("xgboost", "primary", make_xgboost),
            ModelSpec("random_forest", "baseline", make_random_forest),
        ]
    raise ValueError(f"unknown modality: {modality!r}")


def pos_weight(y: np.ndarray) -> float:
    """scale_pos_weight = n_negative / n_positive (>= 1e-6 guard)."""
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


# ---------------------------------------------------------------------------
# evaluation engine
# ---------------------------------------------------------------------------
@dataclass
class ModelEvaluation:
    spec: ModelSpec
    oof_prob: np.ndarray
    fold_metrics: list[dict] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    fold_group_overlap: list[set] = field(default_factory=list)


def evaluate_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    cv,
    seed: int,
    threshold: float = 0.5,
    check_group_leakage: bool = True,
) -> ModelEvaluation:
    """Out-of-fold probability evaluation for one model spec."""
    y = np.asarray(y, dtype=int)
    oof_prob = np.full(len(y), np.nan)
    fold_metrics: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        iter_folds(cv, X, y, groups, check_group_leakage=check_group_leakage), start=1
    ):
        pipeline = spec.factory(pos_weight(y[train_idx]), seed)
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        prob = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        oof_prob[test_idx] = prob
        fold_metrics.append({
            "fold": fold_idx,
            **_compute_metrics(y[test_idx], prob, threshold=threshold),
        })

    if np.isnan(oof_prob).any():
        raise RuntimeError(f"{spec.name}: some samples never received an OOF probability")

    return ModelEvaluation(
        spec=spec,
        oof_prob=oof_prob,
        fold_metrics=fold_metrics,
        aggregate=aggregate_fold_metrics(fold_metrics),
    )


# ---------------------------------------------------------------------------
# modality training entry point
# ---------------------------------------------------------------------------
def train_modality(
    modality: str,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    cfg: Config,
    dataset_info: Mapping[str, object] | None = None,
    threshold: float = 0.5,
) -> dict:
    """Full training cycle for one modality.

    1. cross-validated evaluation of every model spec (OOF probabilities)
    2. refit of every model on the full dataset -> joblib artifacts
    3. metrics JSON + OOF probability CSV for the fusion stage
    """
    seed = cfg.seed
    cv_cfg = cfg.section("cv")
    strategy = default_strategy_for(modality)
    n_folds = int(cv_cfg.get(f"{modality}_folds", 5))
    cv = make_cv(strategy, n_splits=n_folds, seed=seed)

    specs = build_model_specs(modality, seed)
    # voice is record-level by spec (subjects span folds by design); grouped
    # modalities keep the hard leakage assertion
    check_leakage = strategy != "stratified"
    evaluations = [
        evaluate_model(spec, X, y, groups, cv, seed=seed, threshold=threshold,
                       check_group_leakage=check_leakage)
        for spec in specs
    ]
    primary_eval = next(e for e in evaluations if e.spec.role == "primary")

    # refit all models on the full dataset and persist
    artifacts_dir = cfg.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for evaluation in evaluations:
        pipeline = evaluation.spec.factory(pos_weight(y), seed)
        pipeline.fit(X, y)
        joblib.dump(pipeline, artifacts_dir / f"{modality}_{evaluation.spec.name}.joblib")

    # OOF probability table for the fusion stage
    oof_frame = pd.DataFrame({"y": y})
    for evaluation in evaluations:
        oof_frame[evaluation.spec.name] = evaluation.oof_prob
    if groups is not None:
        oof_frame["group"] = np.asarray(groups)
    oof_path = cfg.metrics_dir / f"oof_{modality}.csv"
    cfg.metrics_dir.mkdir(parents=True, exist_ok=True)
    oof_frame.to_csv(oof_path, index_label="row_id")

    primary_name = primary_eval.spec.name
    report = {
        "modality": modality,
        "split_strategy": strategy,
        "n_folds": n_folds if strategy != "logo" else "leave-one-group-out",
        "threshold": threshold,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": [str(c) for c in X.columns],
        "n_groups": int(pd.unique(np.asarray(groups)).size) if groups is not None else None,
        "primary_model": primary_name,
        "primary_oof_roc_auc": primary_eval.aggregate.get("roc_auc_mean"),
        "dataset_info": dict(dataset_info or {}),
        "models": {
            evaluation.spec.name: {
                "role": evaluation.spec.role,
                "aggregate": evaluation.aggregate,
                "fold_metrics": evaluation.fold_metrics,
                "artifact": str(artifacts_dir / f"{modality}_{evaluation.spec.name}.joblib"),
            }
            for evaluation in evaluations
        },
        "oof_probabilities": str(oof_path),
    }
    save_metrics_report(f"{modality}_metrics", report, cfg.metrics_dir)
    return report


__all__ = [
    "ModelSpec",
    "ModelEvaluation",
    "build_model_specs",
    "evaluate_model",
    "train_modality",
    "make_xgboost",
    "make_random_forest",
    "make_logistic_regression",
    "make_svm",
    "pos_weight",
]
