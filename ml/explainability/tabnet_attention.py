"""TabNet explainability via its sparse attention masks.

pytorch-tabnet's ``explain`` returns per-step decision masks; the standard
aggregation (mean of squared masks across steps, as exposed by
``feature_importances_`` for global importances) yields per-feature
attributions that we additionally group into *modality* contributions —
each modality block is its probability input plus its missing indicator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier

from ml.config import Config
from ml.models.fusion import MODALITIES

TABNET_ARTIFACT = "fusion_tabnet"


def load_tabnet(cfg: Config) -> TabNetClassifier:
    """Load the persisted fusion TabNet artifact."""
    path = Path(cfg.artifacts_dir) / f"{TABNET_ARTIFACT}.zip"
    if not path.exists():
        raise FileNotFoundError(f"TabNet artifact missing: {path} (run the fusion stage first)")
    model = TabNetClassifier()
    model.load_model(str(path))
    return model


def instance_attention(model: TabNetClassifier, X_row: np.ndarray) -> dict[str, float]:
    """Attention-based attribution for one meta-feature row.

    ``X_row``: shape (1, n_features) or (n_features,) numpy array in the
    model's training column order.  Returns the aggregated mask values.
    """
    X_row = np.asarray(X_row, dtype=float)
    if X_row.ndim == 1:
        X_row = X_row[None, :]
    explanations, _ = model.explain(X_row)
    agg = np.asarray(explanations)[0]
    return {f"meta_{i}": float(v) for i, v in enumerate(agg)}


def attention_with_names(model: TabNetClassifier, X_row: np.ndarray,
                         feature_names: list[str]) -> dict[str, float]:
    """Instance attention keyed by the actual meta-feature names."""
    X_row = np.asarray(X_row, dtype=float)
    if X_row.ndim == 1:
        X_row = X_row[None, :]
    explanations, _ = model.explain(X_row)
    values = np.asarray(explanations)[0]
    return {str(name): float(v) for name, v in zip(feature_names, values)}


def modality_contributions(attention: Mapping[str, float],
                           feature_names: list[str]) -> dict[str, float]:
    """Aggregate per-feature attention into per-modality contributions.

    Each modality block = ``<modality>_prob`` + ``<modality>_missing``;
    contributions are normalised to sum to 1 over available modalities.
    """
    by_modality = {m: 0.0 for m in MODALITIES}
    for name, value in attention.items():
        for m in MODALITIES:
            if name == f"{m}_prob" or name == f"{m}_missing":
                by_modality[m] += float(value)
                break
    total = sum(by_modality.values())
    if total <= 0:
        return {m: 0.0 for m in MODALITIES}
    return {m: float(v / total) for m, v in by_modality.items()}


def global_feature_importance(model: TabNetClassifier,
                              feature_names: list[str]) -> dict[str, float]:
    """Training-set aggregated attention importances keyed by feature name."""
    values = np.asarray(model.feature_importances_)
    return {str(name): float(v) for name, v in zip(feature_names, values)}


__all__ = [
    "load_tabnet",
    "instance_attention",
    "attention_with_names",
    "modality_contributions",
    "global_feature_importance",
    "TABNET_ARTIFACT",
]
