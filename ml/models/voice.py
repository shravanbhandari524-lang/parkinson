"""Voice modality training (UCI Parkinson's dataset).

Primary model: XGBoost.  Baselines: Logistic Regression, Random Forest.
Split: StratifiedKFold (recording-level, per project spec).
"""

from __future__ import annotations

import numpy as np

from ml.config import Config, get_config
from ml.data.loaders import load_voice
from ml.models.common import train_modality


def train_voice(cfg: Config | None = None) -> dict:
    """Cross-validate, refit and persist all voice models; return the report."""
    cfg = cfg or get_config()
    dataset = load_voice(cfg)
    return train_modality(
        modality="voice",
        X=dataset.X,
        y=dataset.y.to_numpy(),
        groups=dataset.groups.to_numpy(),
        cfg=cfg,
        dataset_info={
            "source": str(cfg.voice_path),
            "description": dataset.description,
            "class_counts": dataset.class_counts,
            "n_voice_subjects": int(dataset.metadata["voice_subject"].nunique()),
            "note": (
                "Recording-level StratifiedKFold per project spec; the UCI dataset "
                "contains multiple recordings per subject, so these numbers are "
                "optimistic relative to a subject-level split."
            ),
        },
    )


def load_voice_training_frame(cfg: Config | None = None):
    """Convenience accessor returning ``(X, y, groups)`` for reuse/tests."""
    dataset = load_voice(cfg or get_config())
    return dataset.X, dataset.y.to_numpy(dtype=int), dataset.groups.to_numpy()


__all__ = ["train_voice", "load_voice_training_frame"]
