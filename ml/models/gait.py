"""Gait modality training (PhysioNet VGRF time series).

Pipeline: per-record time/frequency feature extraction
(:mod:`ml.features.gait`) -> XGBoost (primary) / Random Forest (baseline).
Split: GroupKFold on subject ids — records from the same subject never
straddle train and validation folds.
"""

from __future__ import annotations

import pandas as pd

from ml.config import Config, get_config
from ml.data.loaders import load_gait
from ml.features.gait import featurize_gait_records
from ml.models.common import train_modality


def build_gait_training_frame(cfg: Config | None = None,
                              show_progress: bool = True) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the record table and merge engineered VGRF features per record."""
    cfg = cfg or get_config()
    dataset = load_gait(cfg)
    features = featurize_gait_records(dataset, show_progress=show_progress)
    record_ids = dataset.X["record_id"].astype(str)
    X = features.reindex(record_ids)
    if X.isna().all(axis=1).any():
        bad = X.index[X.isna().all(axis=1)].tolist()
        raise RuntimeError(f"gait records produced no features: {bad}")
    return X, dataset.y, dataset.groups


def train_gait(cfg: Config | None = None) -> dict:
    """Featurize, cross-validate, refit and persist all gait models."""
    cfg = cfg or get_config()
    X, y, groups = build_gait_training_frame(cfg)
    return train_modality(
        modality="gait",
        X=X,
        y=y.to_numpy(),
        groups=groups.to_numpy(),
        cfg=cfg,
        dataset_info={
            "source": str(cfg.gait_path),
            "description": (
                "PhysioNet Gait in Parkinson's Disease; 100 Hz VGRF (8 left + 8 right "
                "sensors), per-record time/frequency features"
            ),
            "class_counts": {int(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "n_subjects": int(groups.nunique()),
            "feature_names": list(X.columns),
        },
    )


def load_gait_training_frame(cfg: Config | None = None):
    """Convenience accessor returning ``(X, y, groups)`` for reuse/tests."""
    return build_gait_training_frame(cfg, show_progress=False)


__all__ = ["train_gait", "build_gait_training_frame", "load_gait_training_frame"]
