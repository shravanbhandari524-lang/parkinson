"""Metric suite and JSON report persistence.

All metrics are computed from out-of-fold predictions on the real datasets —
nothing here fabricates or extrapolates numbers.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Full metric suite at a decision threshold.

    sensitivity == recall (of the PD class); specificity is recall of the
    healthy class.  ROC-AUC is ``None`` when a fold contains one class only.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    roc_auc: float | None
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else None
    except ValueError:  # pragma: no cover - defensive
        roc_auc = None

    def _safe(num: int, den: int) -> float:
        return float(num / den) if den > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": _safe(tp, tp + fn),
        "specificity": _safe(tn, tn + fp),
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def aggregate_fold_metrics(fold_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and std across folds for every numeric metric (None values kept as None)."""
    if not fold_metrics:
        return {}
    keys = fold_metrics[0].keys()
    agg: dict[str, Any] = {}
    for key in keys:
        values = [m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))]
        if key == "confusion_matrix" or not values:
            agg[key] = None
            continue
        arr = np.asarray(values, dtype=float)
        agg[f"{key}_mean"] = float(arr.mean())
        agg[f"{key}_std"] = float(arr.std(ddof=1)) if len(values) > 1 else 0.0
    return agg


def library_versions() -> dict[str, str]:
    """Versions of the core libraries (recorded in every metrics report)."""
    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": _optional_version("pandas"),
        "scikit-learn": sklearn.__version__,
    }
    for name in ("xgboost", "shap", "torch", "pytorch_tabnet"):
        version = _optional_version(name)
        if version:
            versions[name] = version
    return versions


def _optional_version(module_name: str) -> str | None:
    try:
        import importlib

        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", None))
    except Exception:  # noqa: BLE001 - version recording must never fail training
        return None


def save_metrics_report(
    name: str,
    payload: Mapping[str, Any],
    metrics_dir: str | Path,
) -> Path:
    """Write ``<name>.json`` with a UTC timestamp and library versions attached."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": library_versions(),
        **json.loads(json.dumps(payload, default=_json_default)),
    }
    path = metrics_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


__all__ = [
    "compute_metrics",
    "aggregate_fold_metrics",
    "library_versions",
    "save_metrics_report",
]
