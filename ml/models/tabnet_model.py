"""Proposed multimodal fusion model: PyTorch TabNet.

TabNet consumes the same score-level meta-dataset as the baseline fusion
combiners (per-modality out-of-fold primary-model probabilities + missing
modality indicators — see :mod:`ml.models.fusion`).  Its sparse attention
masks double as built-in feature attribution, which the explainability
module turns into per-feature and per-modality contributions.

Grouping: early stopping uses a *group-safe* inner split (GroupShuffleSplit
over patients/subjects), and final reported numbers come from the same
outer StratifiedGroupKFold folds as the baselines — no leakage path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from ml.config import Config
from ml.metrics import compute_metrics, save_metrics_report
from ml.metrics.metrics import aggregate_fold_metrics
from ml.models.fusion import MODALITIES, meta_feature_matrix

SEED = 42


def _tabnet(seed: int) -> TabNetClassifier:
    """TabNet with a compact configuration suited to a small meta-feature space."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return TabNetClassifier(
        n_d=16, n_a=16,
        n_steps=4,
        gamma=1.3,
        n_independent=1, n_shared=1,
        lambda_sparse=1e-3,
        clip_value=2.0,
        momentum=0.02,
        optimizer_params=dict(lr=3e-2),
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax",
        seed=seed,
        verbose=0,
        device_name="auto",
    )


def _fit_tabnet(model: TabNetClassifier, X: np.ndarray, y: np.ndarray,
                groups: np.ndarray, seed: int, max_epochs: int = 200,
                patience: int = 25) -> TabNetClassifier:
    """Fit with a group-safe inner validation split for early stopping."""
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(inner.split(X, y, groups=groups))
    model.fit(
        X[tr_idx], y[tr_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=1,  # automatic class balancing (meta-dataset is ~72% positive)
    )
    return model


def evaluate_tabnet_fusion(cfg: Config,
                           meta: pd.DataFrame | None = None,
                           seed: int = SEED,
                           max_epochs: int = 200,
                           patience: int = 25) -> dict:
    """Cross-validate TabNet on the meta-dataset; refit and persist the model.

    Returns ``{"report": ..., "model": ..., "X": ..., "feature_names": ...}``
    so the explainability module can reuse the fitted artifact.
    """
    from ml.models.fusion import build_meta_dataset

    if meta is None:
        meta = build_meta_dataset(cfg)
    X, _, feature_names = meta_feature_matrix(meta)
    y = meta["y"].to_numpy(dtype=int)
    groups = meta["group"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_rows: list[dict] = []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups=groups), start=1):
        model = _tabnet(seed + fold_idx)
        _fit_tabnet(model, X[tr], y[tr], groups[tr], seed + fold_idx,
                    max_epochs=max_epochs, patience=patience)
        prob = model.predict_proba(X[te])[:, 1]
        fold_rows.append({"fold": fold_idx, **compute_metrics(y[te], prob)})

    # refit on the full meta-dataset and persist
    final_model = _tabnet(seed)
    _fit_tabnet(final_model, X, y, groups, seed, max_epochs=max_epochs, patience=patience)
    artifacts_dir = cfg.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(artifacts_dir) / "fusion_tabnet"
    final_model.save_model(str(model_path))

    aggregate = aggregate_fold_metrics(fold_rows)
    report = {
        "role": "proposed_model",
        "model": "TabNetClassifier (pytorch-tabnet)",
        "input": (
            "score-level meta-features: 3 modality OOF probabilities (neutral-filled "
            "0.5 when missing) + 3 missing-modality indicators"
        ),
        "aggregate": aggregate,
        "fold_metrics": fold_rows,
        "artifact_prefix": str(model_path),
        "grouping": (
            "outer StratifiedGroupKFold over patients/subjects; early-stopping "
            "validation split is group-safe (GroupShuffleSplit)"
        ),
    }
    save_metrics_report("tabnet_fusion_metrics", report, cfg.metrics_dir)
    return {"report": report, "model": final_model, "X": X,
            "feature_names": feature_names, "meta": meta}


__all__ = ["evaluate_tabnet_fusion", "_tabnet", "_fit_tabnet", "SEED"]
