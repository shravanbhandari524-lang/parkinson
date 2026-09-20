"""Multimodal prediction API with built-in explainability.

``MultimodalPredictor.predict`` accepts any subset of the three modalities
for one patient and returns::

    {
      "probability": float,                  # PD risk from the fusion model
      "predicted_class": 0 | 1,
      "per_modality_probability": {...},
      "top_features": [...],                 # most influential, per modality
      "feature_contributions": {...},        # signed SHAP per feature
      "modality_contribution": {...},        # TabNet attention, sums to 1
    }

Tree explainability uses SHAP (exact TreeExplainer values); the fusion-level
modality weights come from TabNet's attention masks, or from the fitted
soft-voting weights when the TabNet artifact is unavailable.
"""

from __future__ import annotations

from typing import Mapping

import joblib
import numpy as np
import pandas as pd

from ml.config import Config, get_config
from ml.explainability.shap_explainer import instance_contributions
from ml.explainability.tabnet_attention import (
    attention_with_names,
    modality_contributions,
)
from ml.features.gait import featurize_record
from ml.models.fusion import MODALITIES, NEUTRAL_PROB

META_FEATURE_NAMES = [f"{m}_prob" for m in MODALITIES] + \
                     [f"{m}_missing" for m in MODALITIES]
TOP_K_FEATURES = 10


class MultimodalPredictor:
    """Loads the persisted artifacts once and serves explained predictions."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        self._primary: dict[str, object] = {}
        self._feature_names: dict[str, list[str]] = {}
        self._tabnet = None
        self._voter = None
        self._load_artifacts()

    # ------------------------------------------------------------------ load
    def _load_artifacts(self) -> None:
        import json

        for modality in MODALITIES:
            metrics_path = self.cfg.metrics_dir / f"{modality}_metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(
                    f"{metrics_path} missing — train the modality first (ml/train_all.py)")
            with open(metrics_path, encoding="utf-8") as fh:
                report = json.load(fh)
            primary = report["primary_model"]
            artifact = self.cfg.artifacts_dir / f"{modality}_{primary}.joblib"
            if not artifact.exists():
                raise FileNotFoundError(f"model artifact missing: {artifact}")
            self._primary[modality] = joblib.load(artifact)
            names = report.get("feature_names")
            if not names:
                raise KeyError(f"{metrics_path} lacks feature_names (retrain to add them)")
            self._feature_names[modality] = list(names)

        tabnet_path = self.cfg.artifacts_dir / "fusion_tabnet.zip"
        if tabnet_path.exists():
            from pytorch_tabnet.tab_model import TabNetClassifier

            self._tabnet = TabNetClassifier()
            self._tabnet.load_model(str(tabnet_path))
        else:  # graceful degradation to the weighted-voting baseline
            self._voter = joblib.load(
                self.cfg.artifacts_dir / "fusion_weighted_soft_voting.joblib")

    # ----------------------------------------------------------------- inputs
    @staticmethod
    def _feature_row(modality: str, names: list[str], values: Mapping[str, object] | pd.DataFrame) -> pd.DataFrame:
        """Build a one-row feature frame in training column order."""
        if isinstance(values, pd.DataFrame):
            row = values.iloc[[0]]
        else:
            missing = [n for n in names if n not in values]
            if missing:
                raise ValueError(f"{modality} input missing features: {missing[:5]}")
            row = pd.DataFrame([{n: values[n] for n in names}])
        return row[list(names)]

    def _modality_probability(self, modality: str, row: pd.DataFrame) -> float:
        pipeline = self._primary[modality]
        return float(pipeline.predict_proba(row)[0, 1])

    # ----------------------------------------------------------------- predict
    def predict(
        self,
        voice: Mapping[str, object] | pd.DataFrame | None = None,
        handwriting: Mapping[str, object] | pd.DataFrame | None = None,
        gait: np.ndarray | Mapping[str, object] | None = None,
    ) -> dict:
        """Predict PD risk from any subset of modalities and explain it.

        ``voice``/``handwriting``: mapping of feature name -> value (or a
        one-row DataFrame).  ``gait``: either a raw ``(n_samples, 19)`` VGRF
        array (featurized automatically) or a mapping of engineered feature
        name -> value.
        """
        provided: dict[str, pd.DataFrame] = {}
        if voice is not None:
            provided["voice"] = self._feature_row("voice", self._feature_names["voice"], voice)
        if handwriting is not None:
            provided["handwriting"] = self._feature_row(
                "handwriting", self._feature_names["handwriting"], handwriting)
        if gait is not None:
            if isinstance(gait, np.ndarray):
                gait_features = featurize_record(gait)
            else:
                gait_features = dict(gait)
            provided["gait"] = self._feature_row("gait", self._feature_names["gait"], gait_features)

        if not provided:
            raise ValueError("provide at least one modality (voice/handwriting/gait)")

        per_modality_prob = {
            m: self._modality_probability(m, row) for m, row in provided.items()
        }

        # meta-feature vector for the fusion stage
        probs = [per_modality_prob.get(m, NEUTRAL_PROB) for m in MODALITIES]
        missing = [0.0 if m in provided else 1.0 for m in MODALITIES]
        meta_row = np.array([probs + missing], dtype=float)

        probability = self._fuse(meta_row)
        predicted_class = int(probability >= 0.5)

        return {
            "probability": probability,
            "predicted_class": predicted_class,
            "per_modality_probability": per_modality_prob,
            "top_features": self._top_features(provided),
            "feature_contributions": {
                m: instance_contributions(self._primary[m], row)
                for m, row in provided.items()
            },
            "modality_contribution": self._modality_contribution(meta_row),
            "fusion_model": "tabnet" if self._tabnet is not None else "weighted_soft_voting",
        }

    # ------------------------------------------------------------- explainers
    def _fuse(self, meta_row: np.ndarray) -> float:
        if self._tabnet is not None:
            return float(self._tabnet.predict_proba(meta_row)[0, 1])
        probs = meta_row[0, : len(MODALITIES)].copy()
        probs[meta_row[0, len(MODALITIES):] == 1.0] = np.nan
        return float(self._voter.predict_proba(probs[None, :])[0, 1])

    def _top_features(self, provided: dict[str, pd.DataFrame]) -> list[dict]:
        scored: list[dict] = []
        for modality, row in provided.items():
            contributions = instance_contributions(self._primary[modality], row)
            for feature, value in contributions.items():
                scored.append({
                    "modality": modality,
                    "feature": feature,
                    "contribution": value,
                    "abs_contribution": abs(value),
                })
        scored.sort(key=lambda item: -item["abs_contribution"])
        top = [{k: v for k, v in item.items() if k != "abs_contribution"}
               for item in scored[:TOP_K_FEATURES]]
        return top

    def _modality_contribution(self, meta_row: np.ndarray) -> dict[str, float]:
        if self._tabnet is not None:
            attention = attention_with_names(
                self._tabnet, meta_row, META_FEATURE_NAMES)
            return modality_contributions(attention, META_FEATURE_NAMES)
        # fallback: fitted voting weights over the available modalities
        weights = {m: float(w) for m, w in zip(MODALITIES, self._voter.weights)}
        available = {m: weights[m] for m in MODALITIES if meta_row[0, len(MODALITIES) + MODALITIES.index(m)] == 0.0}
        total = sum(available.values()) or 1.0
        return {m: float(available.get(m, 0.0) / total) for m in MODALITIES}


__all__ = ["MultimodalPredictor", "META_FEATURE_NAMES"]
