"""SHAP explainability for the tree-based modality models.

Works on the fitted scikit-learn pipelines: the preprocessing step is applied
first, then ``shap.TreeExplainer`` produces exact Shapley values for the
XGBoost / Random Forest estimators.  Contributions are reported in the
model's native output space (log-odds for XGBoost, probability space for the
random forest), always for the PD class (label 1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import shap

from ml.config import Config


def make_tree_explainer(pipeline) -> shap.TreeExplainer:
    """Build a TreeExplainer for a pipeline's final estimator."""
    estimator = pipeline.steps[-1][1]
    return shap.TreeExplainer(estimator)


def _class1_shap_values(explainer: shap.TreeExplainer, X_trans: np.ndarray) -> np.ndarray:
    """Shapley values for class 1 across shap versions/output shapes."""
    values = explainer.shap_values(X_trans)
    if isinstance(values, list):  # sklearn forests: [class0, class1]
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        values = values[:, :, 1]
    return values


def instance_contributions(pipeline, X_row: pd.DataFrame) -> dict[str, float]:
    """Signed per-feature SHAP contributions for one prediction (PD class).

    ``X_row`` must be a one-row DataFrame with the pipeline's training
    columns; the pipeline's preprocessor is applied before explaining.
    """
    if len(X_row) != 1:
        raise ValueError("explain exactly one row at a time")
    pre = pipeline.steps[0][1]
    estimator_name, estimator = pipeline.steps[-1]
    X_trans = pre.transform(X_row)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    explainer = shap.TreeExplainer(estimator)
    values = _class1_shap_values(explainer, np.asarray(X_trans, dtype=float))[0]
    names = _transformed_feature_names(pre, X_row, estimator_name)
    return {str(name): float(value) for name, value in zip(names, values)}


def _transformed_feature_names(pre, X_row: pd.DataFrame, estimator_name: str) -> list[str]:
    """Feature names after preprocessing (numeric passthrough here)."""
    try:
        names = pre.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:  # noqa: BLE001 - fall back to input names
        return [str(c) for c in X_row.columns]


def global_importance(pipeline, X: pd.DataFrame, max_rows: int = 500) -> dict[str, float]:
    """Mean |SHAP| per feature over a background sample of ``X``."""
    background = X.iloc[: min(len(X), max_rows)]
    pre = pipeline.steps[0][1]
    X_trans = pre.transform(background)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    explainer = make_tree_explainer(pipeline)
    values = np.abs(_class1_shap_values(explainer, np.asarray(X_trans, dtype=float))).mean(axis=0)
    names = _transformed_feature_names(pre, background, pipeline.steps[-1][0])
    return {str(name): float(value) for name, value in zip(names, values)}


def summarize_modality(modality: str, cfg: Config,
                       background_rows: int = 500) -> dict:
    """Global SHAP summary for a modality's primary model, saved to JSON.

    Uses the real training data (via the modality loader / featurizer) as the
    background distribution and the persisted artifact as the model.
    """
    import joblib

    artifact = cfg.artifacts_dir / f"{modality}_primary.joblib"
    if not artifact.exists():
        from ml.models.common import build_model_specs

        primary = next(s.name for s in build_model_specs(modality, cfg.seed) if s.role == "primary")
        artifact = cfg.artifacts_dir / f"{modality}_{primary}.joblib"
    if not artifact.exists():
        raise FileNotFoundError(f"no trained model for {modality}: {artifact}")
    pipeline = joblib.load(artifact)

    X = _modality_background(modality, cfg, background_rows)
    importance = global_importance(pipeline, X, max_rows=background_rows)
    summary = {
        "modality": modality,
        "method": "SHAP TreeExplainer (mean |SHAP| for the PD class)",
        "model_artifact": str(artifact),
        "background_rows": int(len(X)),
        "feature_importance": dict(sorted(importance.items(), key=lambda kv: -abs(kv[1]))),
    }
    out = cfg.metrics_dir / f"shap_summary_{modality}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    summary["saved_to"] = str(out)
    return summary


def _modality_background(modality: str, cfg: Config, rows: int) -> pd.DataFrame:
    """Real background rows for the given modality's primary model."""
    if modality == "voice":
        from ml.models.voice import load_voice_training_frame

        X, _, _ = load_voice_training_frame(cfg)
    elif modality == "handwriting":
        from ml.models.handwriting import load_handwriting_training_frame

        X, _, _ = load_handwriting_training_frame(cfg)
    elif modality == "gait":
        from ml.models.gait import load_gait_training_frame

        X, _, _ = load_gait_training_frame(cfg)
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown modality: {modality!r}")
    return X.iloc[: min(len(X), rows)]


__all__ = [
    "make_tree_explainer",
    "instance_contributions",
    "global_importance",
    "summarize_modality",
]
