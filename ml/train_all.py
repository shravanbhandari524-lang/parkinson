"""End-to-end training pipeline for the multimodal PD risk framework.

Stages (in order, each persisted under ``ml/metrics`` and ``ml/artifacts``):

1. voice      — XGBoost + LR/RF baselines, StratifiedKFold
2. handwriting— RF + SVM/LR baselines, GroupKFold by patient
3. gait       — VGRF featurization + XGBoost/RF, GroupKFold by subject
4. fusion     — weighted soft voting + LR stacking (baselines)
5. tabnet     — proposed TabNet multimodal fusion
6. explain    — SHAP summaries for the tree primaries

Usage::

    ml/.venv/Scripts/python.exe ml/train_all.py                # everything
    ml/.venv/Scripts/python.exe ml/train_all.py --skip voice   # skip a stage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# allow `python ml/train_all.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "  -  "


def _print_modality(report: dict) -> None:
    print(f"\n=== {report['modality']} ({report['split_strategy']}, "
          f"{report['n_samples']} samples, {report['n_features']} features) ===")
    for name, model in report["models"].items():
        agg = model["aggregate"]
        print(f"  {name:22s} role={model['role']:8s} "
              f"acc={_fmt(agg['accuracy_mean'])} sens={_fmt(agg['sensitivity_mean'])} "
              f"spec={_fmt(agg['specificity_mean'])} auc={_fmt(agg['roc_auc_mean'])}")


def _print_fusion(report: dict) -> None:
    print("\n=== fusion (score-level, grouped CV) ===")
    for name, model in report["baseline_fusion"].items():
        agg = model["aggregate"]
        print(f"  {name:34s} acc={_fmt(agg['accuracy_mean'])} "
              f"sens={_fmt(agg['sensitivity_mean'])} spec={_fmt(agg['specificity_mean'])} "
              f"auc={_fmt(agg['roc_auc_mean'])}")
    if report.get("tabnet"):
        agg = report["tabnet"]["report"]["aggregate"]
        print(f"  {'tabnet_fusion (proposed)':34s} acc={_fmt(agg['accuracy_mean'])} "
              f"sens={_fmt(agg['sensitivity_mean'])} spec={_fmt(agg['specificity_mean'])} "
              f"auc={_fmt(agg['roc_auc_mean'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", action="append", default=[],
                        choices=["voice", "handwriting", "gait", "fusion", "explain"],
                        help="skip a pipeline stage (repeatable)")
    parser.add_argument("--tabnet-epochs", type=int, default=200)
    parser.add_argument("--tabnet-patience", type=int, default=25)
    args = parser.parse_args(argv)
    skip = set(args.skip)

    started = time.time()
    from ml.config import get_config

    cfg = get_config()
    print(f"data dir: {cfg.data_dir.resolve()}")
    print(f"artifacts: {cfg.artifacts_dir.resolve()}")
    print(f"metrics: {cfg.metrics_dir.resolve()}")

    if "voice" not in skip:
        from ml.models.voice import train_voice

        _print_modality(train_voice(cfg))

    if "handwriting" not in skip:
        from ml.models.handwriting import train_handwriting

        _print_modality(train_handwriting(cfg))

    if "gait" not in skip:
        from ml.models.gait import train_gait

        _print_modality(train_gait(cfg))

    if "fusion" not in skip:
        from ml.models.fusion import evaluate_fusion

        fusion_report = evaluate_fusion(cfg, include_tabnet=True)
        # trim the tabnet block for printing
        fusion_json = json.loads(json.dumps(fusion_report, default=str))
        _print_fusion(fusion_json)
        _ = args.tabnet_epochs  # epochs/patience are handled inside the CV stage

    if "explain" not in skip:
        from ml.explainability.shap_explainer import summarize_modality

        for modality in ("voice", "handwriting", "gait"):
            summary = summarize_modality(modality, cfg)
            top = list(summary["feature_importance"].items())[:5]
            pretty = ", ".join(f"{k}={v:.3f}" for k, v in top)
            print(f"  SHAP {modality:11s} top-5: {pretty}")

    print(f"\ndone in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
