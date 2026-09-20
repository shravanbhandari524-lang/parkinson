"""TabNet fusion smoke tests and integration tests for the inference API.

The TabNet smoke test uses a synthetic meta-frame and a temp artifacts dir —
it never touches the real artifacts.  The inference integration tests are
gated on the real artifacts existing (they run after ``ml/train_all.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.config import Config
from ml.models.fusion import MODALITIES
from ml.models.tabnet_model import evaluate_tabnet_fusion
from ml.tests.test_fusion import _synthetic_meta


@pytest.fixture(scope="module")
def tmp_cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    return Config({
        "data_dir": "Parkinson",
        "voice": {"path": "Voice/parkinsons.data"},
        "handwriting": {"spiral_path": "x", "meander_path": "y"},
        "gait": {"path": "z"},
        "seed": 42,
        "artifacts_dir": str(out),
        "metrics_dir": str(tmp_path_factory.mktemp("metrics")),
    })


def test_tabnet_smoke_on_synthetic_meta(tmp_cfg):
    meta = _synthetic_meta(n=60, seed=3)
    result = evaluate_tabnet_fusion(tmp_cfg, meta=meta, max_epochs=5, patience=3)
    report = result["report"]
    assert report["aggregate"]["n_samples_mean"] == pytest.approx(12)
    assert result["X"].shape == (60, 6)
    assert len(result["model"].feature_importances_) == 6
    assert (tmp_cfg.artifacts_dir / "fusion_tabnet.zip").exists()


def test_modality_contributions_grouping(tmp_cfg):
    from ml.explainability.tabnet_attention import (
        attention_with_names,
        load_tabnet,
        modality_contributions,
    )

    model = load_tabnet(tmp_cfg)
    names = [f"{m}_prob" for m in MODALITIES] + [f"{m}_missing" for m in MODALITIES]
    X_row = np.array([[0.7, 0.5, 0.5, 0.0, 1.0, 1.0]])  # voice-only row
    attention = attention_with_names(model, X_row, names)
    contrib = modality_contributions(attention, names)
    assert set(contrib) == set(MODALITIES)
    assert sum(contrib.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in contrib.values())


# ---------------------------------------------------------------------------
# integration (real artifacts; skipped until ml/train_all.py has run)
# ---------------------------------------------------------------------------
ARTIFACTS_MISSING = "artifacts not trained yet — run ml/train_all.py first"


def _artifacts_ready(cfg) -> bool:
    return (cfg.artifacts_dir / "fusion_tabnet.zip").exists() and \
        all((cfg.artifacts_dir / f"{m}_primary.joblib").exists() or True for m in MODALITIES)


@pytest.mark.integration
def test_multimodal_predictor_voice_only():
    from ml.config import get_config
    from ml.inference import MultimodalPredictor
    from ml.models.voice import load_voice_training_frame

    cfg = get_config()
    if not (cfg.artifacts_dir / "fusion_tabnet.zip").exists():
        pytest.skip(ARTIFACTS_MISSING)
    X, y, _ = load_voice_training_frame(cfg)
    predictor = MultimodalPredictor(cfg)
    row = X.iloc[0].to_dict()
    result = predictor.predict(voice=row)
    assert set(result) >= {"probability", "predicted_class", "top_features",
                           "feature_contributions", "modality_contribution"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["predicted_class"] in (0, 1)
    assert result["top_features"] and "feature" in result["top_features"][0]
    assert sum(result["modality_contribution"].values()) == pytest.approx(1.0)
    assert "voice" in result["per_modality_probability"]


@pytest.mark.integration
def test_multimodal_predictor_gait_raw_record():
    from ml.config import get_config
    from ml.data.loaders import load_gait_record, list_gait_records
    from ml.inference import MultimodalPredictor

    cfg = get_config()
    if not (cfg.artifacts_dir / "fusion_tabnet.zip").exists():
        pytest.skip(ARTIFACTS_MISSING)
    records = list_gait_records(cfg.gait_path)
    record = load_gait_record(records.iloc[0]["path"])
    predictor = MultimodalPredictor(cfg)
    result = predictor.predict(gait=record)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["feature_contributions"]["gait"]
