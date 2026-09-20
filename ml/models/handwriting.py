"""Handwriting modality training (HandPD spiral + meander).

Primary model: Random Forest.  Baselines: SVM (RBF), Logistic Regression.
Split: GroupKFold on patient groups (CLASS_TYPE + ID_PATIENT — the HandPD
patient numbering restarts per class, so the class is part of the group key).
"""

from __future__ import annotations

from ml.config import Config, get_config
from ml.data.loaders import load_handwriting
from ml.models.common import train_modality


def train_handwriting(cfg: Config | None = None) -> dict:
    """Cross-validate, refit and persist all handwriting models; return the report."""
    cfg = cfg or get_config()
    dataset = load_handwriting(cfg)
    return train_modality(
        modality="handwriting",
        X=dataset.X,
        y=dataset.y.to_numpy(),
        groups=dataset.groups.to_numpy(),
        cfg=cfg,
        dataset_info={
            "source": [str(cfg.handwriting_spiral_path), str(cfg.handwriting_meander_path)],
            "description": dataset.description,
            "class_counts": dataset.class_counts,
            "n_patient_groups": dataset.n_groups,
            "exam_types": sorted(dataset.metadata["exam_type"].unique().tolist()),
            "grouping_note": (
                "Groups are (CLASS_TYPE, ID_PATIENT): healthy ids run 1-18 and PD ids "
                "run 1-37 (id 4 absent), so the raw ID_PATIENT alone is ambiguous."
            ),
        },
    )


def load_handwriting_training_frame(cfg: Config | None = None):
    """Convenience accessor returning ``(X, y, groups)`` for reuse/tests."""
    dataset = load_handwriting(cfg or get_config())
    return dataset.X, dataset.y.to_numpy(dtype=int), dataset.groups.to_numpy()


__all__ = ["train_handwriting", "load_handwriting_training_frame"]
