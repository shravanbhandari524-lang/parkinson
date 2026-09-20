"""Python ML service (internal, never exposed publicly).

Called only by the Express backend (see backend/src/services/ml.service.js):

    GET  /health                 -> {status, model_version, models_loaded}
    GET  /model-info             -> training metadata + cross-validated metrics
    POST /predict/voice          (multipart field `file`)
    POST /predict/handwriting    (multipart field `file`)
    POST /predict/gait           (multipart field `file`; raw VGRF record or features CSV)
    POST /predict/multimodal     (multipart fields `voice`, `handwriting`, `gait` — any subset)

Every probability/SHAP value is computed live by the trained artifacts via
:class:`ml.inference.MultimodalPredictor`. Nothing is mocked or hard-coded;
if an artifact is missing the API reports the exact missing file instead of
returning a fabricated prediction.
"""

from __future__ import annotations

import csv
import io
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ml.config import get_config
from ml.inference import MultimodalPredictor

MODEL_VERSION = "1.0.0"

VOICE_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# The model's canonical handwriting names use STROKE; the raw HandPD CSVs use
# ET — both spellings are accepted on input.
HANDWRITING_FEATURES = [
    "RMS", "MAX_BETWEEN_STROKE_HT", "MIN_BETWEEN_STROKE_HT",
    "STD_DEVIATION_STROKE_HT", "MRT", "MAX_HT", "MIN_HT", "STD_HT",
    "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_STROKE_HT",
]
_HANDWRITING_ALIASES = {
    name: name.replace("ET_HT", "STROKE_HT")
    for name in (
        "MAX_BETWEEN_ET_HT", "MIN_BETWEEN_ET_HT", "STD_DEVIATION_ET_HT",
        "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT",
    )
}

MODALITIES = ("voice", "handwriting", "gait")


def risk_band(probability: float) -> str:
    """Plan thresholds: <0.35 low, 0.35-0.65 borderline, >0.65 high."""
    if probability < 0.35:
        return "low"
    if probability > 0.65:
        return "high"
    return "borderline"


# ---------------------------------------------------------------------------
# file parsing (accepts real dataset exports; no synthetic inputs)
# ---------------------------------------------------------------------------
def _split_lines(text: str) -> list[list[str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(422, "uploaded file is empty")
    # PhysioNet records are tab-separated; dataset exports use ',' or ';'
    delimiter = max((",", ";", "\t"), key=lines[0].count)
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)
    return [row for row in reader]


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def parse_voice_csv(data: bytes) -> dict[str, float]:
    """One voice sample: header row preferred; UCI column order fallback."""
    rows = _split_lines(data.decode("utf-8-sig", errors="replace"))
    header = [c.strip() for c in rows[0]]
    has_header = any(not _is_number(c) for c in header)
    if has_header:
        names = header
        data_row = rows[1] if len(rows) > 1 else None
        if data_row is None:
            raise HTTPException(422, "voice CSV has a header but no data row")
        row = dict(zip(names, (c.strip() for c in data_row)))
    else:
        # headerless: 22 features in UCI order, or full 24-column UCI row
        values = [c.strip() for c in rows[0]]
        if len(values) == 24:
            values = values[1:-1]  # drop `name` and `status`
        if len(values) != len(VOICE_FEATURES):
            raise HTTPException(
                422,
                f"headerless voice file must have 22 (or 24 UCI) columns, got {len(values)}",
            )
        row = dict(zip(VOICE_FEATURES, values))

    missing = [f for f in VOICE_FEATURES if f not in row or row[f] == ""]
    if missing:
        raise HTTPException(422, f"voice input missing required features: {missing}")
    try:
        return {f: float(row[f]) for f in VOICE_FEATURES}
    except ValueError as exc:
        raise HTTPException(422, f"voice input has a non-numeric value: {exc}") from exc


HANDWRITING_ALIASES = _HANDWRITING_ALIASES  # public alias for readability


def parse_handwriting_csv(data: bytes) -> dict[str, float]:
    """One HandPD exam row; accepts STROKE (canonical) or ET (raw CSV) names."""
    rows = _split_lines(data.decode("utf-8-sig", errors="replace"))
    header = [c.strip() for c in rows[0]]
    has_header = any(not _is_number(c) for c in header)
    if has_header:
        names = [HANDWRITING_ALIASES.get(h, h) for h in header]
        if len(rows) < 2:
            raise HTTPException(422, "handwriting CSV has a header but no data row")
        row = dict(zip(names, (c.strip() for c in rows[1])))
    else:
        values = [c.strip() for c in rows[0]]
        if len(values) == 17:  # full HandPD row: first 8 columns are metadata
            values = values[8:]
        if len(values) != len(HANDWRITING_FEATURES):
            raise HTTPException(
                422,
                f"headerless handwriting file must have 9 (or 17 HandPD) columns, got {len(values)}",
            )
        row = dict(zip(HANDWRITING_FEATURES, values))

    missing = [f for f in HANDWRITING_FEATURES if f not in row or row[f] == ""]
    if missing:
        raise HTTPException(422, f"handwriting input missing required features: {missing}")
    try:
        return {f: float(row[f]) for f in HANDWRITING_FEATURES}
    except ValueError as exc:
        raise HTTPException(422, f"handwriting input has a non-numeric value: {exc}") from exc


def parse_gait_file(data: bytes, required_features: list[str]) -> np.ndarray | dict[str, float]:
    """Raw VGRF record (19 numeric columns, any delimiter) or a features CSV."""
    text = data.decode("utf-8-sig", errors="replace")
    rows = _split_lines(text)
    header = [c.strip() for c in rows[0]]
    if any(not _is_number(c) for c in header):
        row = dict(zip(header, (c.strip() for c in (rows[1] if len(rows) > 1 else []))))
        missing = [f for f in required_features if f not in row or row[f] == ""]
        if missing:
            raise HTTPException(422, f"gait feature file missing required features: {missing}")
        try:
            return {f: float(row[f]) for f in required_features}
        except ValueError as exc:
            raise HTTPException(422, f"gait feature file has a non-numeric value: {exc}") from exc

    try:
        matrix = np.array([[float(c) for c in row] for row in rows], dtype=float)
    except ValueError as exc:
        raise HTTPException(422, f"gait record contains a non-numeric value: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[1] != 19:
        raise HTTPException(
            422,
            f"raw gait record must be a numeric table with 19 columns "
            f"(time, 8 left VGRF, 8 right VGRF, L/R total force); got shape {matrix.shape}",
        )
    return matrix


# ---------------------------------------------------------------------------
# response builder (matches backend Zod contract + frontend normaliser)
# ---------------------------------------------------------------------------
def _top_feature_items(contributions: Mapping[str, float],
                       values: Mapping[str, float],
                       modality: str, limit: int = 10) -> list[dict]:
    ranked = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:limit]
    return [{
        "modality": modality,
        "feature": feature,
        "contribution": float(value),
        "shap": float(value),
        "value": float(values.get(feature, "nan")) if feature in values else None,
    } for feature, value in ranked]


def build_response(predictor: MultimodalPredictor, result: dict) -> dict:
    probability = result["probability"]
    attention = result.get("meta_attention") or {}

    explanations: dict[str, list] = {}
    for modality in MODALITIES:
        contributions = result["feature_contributions"].get(modality)
        if contributions is None:
            continue
        explanations[modality] = _top_feature_items(
            contributions, result["inputs"].get(modality, {}), modality)

    meta_names = [f"{m}_prob" for m in MODALITIES] + [f"{m}_missing" for m in MODALITIES]
    explanations["multimodal"] = [{
        "feature": name.replace("_prob", ".probability").replace("_missing", ".missing"),
        "shap": float(attention.get(name, 0.0)),
        "value": float(result["meta_values"].get(name)) if name in result.get("meta_values", {}) else None,
    } for name in meta_names]
    explanations["multimodal"] = [e for e in explanations["multimodal"] if e["shap"] != 0.0] or \
        explanations["multimodal"]

    response = {
        "model_version": MODEL_VERSION,
        "timestamp": result.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "risk_band": risk_band(probability),
        "overall_probability": probability,
        "fusion": {
            "method": "TabNet" if result.get("fusion_model") == "tabnet" else "weighted_soft_voting",
            "probability": probability,
            "contribution": None,
        },
        "modalities": {
            modality: {
                "available": modality in result["per_modality_probability"],
                "probability": result["per_modality_probability"].get(modality),
                "contribution": result["modality_contribution"].get(modality),
            }
            for modality in MODALITIES
        },
        "multimodal": {
            "probability": probability,
            "top_features": result["top_features"],
        },
        "explanations": explanations,
    }
    for modality in MODALITIES:
        contributions = result["feature_contributions"].get(modality)
        if contributions is None:
            continue
        response[modality] = {
            "probability": result["per_modality_probability"][modality],
            "top_features": _top_feature_items(
                contributions, result["inputs"].get(modality, {}), modality),
        }
    return response


# ---------------------------------------------------------------------------
# app state
# ---------------------------------------------------------------------------
def _artifact_report() -> tuple[dict[str, bool], list[str]]:
    """Which model files exist and exactly which are missing."""
    cfg = get_config()
    expected = {
        "voice_xgboost": "voice_xgboost.joblib",
        "handwriting_random_forest": "handwriting_random_forest.joblib",
        "gait_xgboost": "gait_xgboost.joblib",
        "fusion_tabnet": "fusion_tabnet.zip",
        "fusion_weighted_soft_voting": "fusion_weighted_soft_voting.joblib",
    }
    loaded = {}
    missing = []
    for key, filename in expected.items():
        path = Path(cfg.artifacts_dir) / filename
        loaded[key] = path.exists()
        if not path.exists():
            missing.append(str(path))
    return loaded, missing


_predictor: MultimodalPredictor | None = None
_load_error: str | None = None


def get_predictor() -> MultimodalPredictor:
    if _predictor is None:
        raise HTTPException(503, f"ML models are not loaded: {_load_error}")
    return _predictor


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _predictor, _load_error
    try:
        _predictor = MultimodalPredictor()
        _load_error = None
    except Exception as exc:  # noqa: BLE001 - report exact missing artifact
        _predictor = None
        _load_error = str(exc)
    yield


app = FastAPI(title="Parkinson Multimodal ML Service", version=MODEL_VERSION,
              lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # internal service; Express is the public gateway
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    loaded, missing = _artifact_report()
    status = "ok" if _predictor is not None and not missing else "degraded"
    body = {
        "status": status,
        "model_version": MODEL_VERSION,
        "models_loaded": loaded,
    }
    if missing:
        body["missing_artifacts"] = missing
    if _load_error:
        body["detail"] = _load_error
    return body


@app.get("/model-info")
def model_info():
    cfg = get_config()
    algorithms = {}
    for modality, metrics_name in (("voice", "voice_metrics"),
                                   ("handwriting", "handwriting_metrics"),
                                   ("gait", "gait_metrics")):
        path = cfg.metrics_dir / f"{metrics_name}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
        primary = report.get("models", {}).get(report.get("primary_model", ""), {})
        agg = primary.get("aggregate", {})
        algorithms[modality] = {
            "primary_model": report.get("primary_model"),
            "split_strategy": report.get("split_strategy"),
            "n_samples": report.get("n_samples"),
            "roc_auc_mean": agg.get("roc_auc_mean"),
            "accuracy_mean": agg.get("accuracy_mean"),
        }
    trained_at = None
    fusion_path = cfg.metrics_dir / "fusion_metrics.json"
    fusion_info = None
    if fusion_path.exists():
        with open(fusion_path, encoding="utf-8") as fh:
            fusion = json.load(fh)
        trained_at = fusion.get("created_utc")
        tabnet = (fusion.get("tabnet") or {}).get("report", {})
        fusion_info = {
            "proposed_model": "TabNet",
            "tabnet_roc_auc_mean": (tabnet.get("aggregate") or {}).get("roc_auc_mean"),
            "note": fusion.get("novelty_note"),
        }
    return {
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "algorithms": algorithms,
        "fusion": fusion_info,
        "disclaimer": "Research prototype — not a medical device; not for diagnosis.",
    }


@app.post("/predict/voice")
def predict_voice(file: UploadFile = File(...)):
    row = parse_voice_csv(file.file.read())
    result = get_predictor().predict(voice=row)
    return build_response(get_predictor(), result)


@app.post("/predict/handwriting")
def predict_handwriting(file: UploadFile = File(...)):
    row = parse_handwriting_csv(file.file.read())
    result = get_predictor().predict(handwriting=row)
    return build_response(get_predictor(), result)


@app.post("/predict/gait")
def predict_gait(file: UploadFile = File(...)):
    predictor = get_predictor()
    parsed = parse_gait_file(file.file.read(), predictor._feature_names["gait"])
    result = predictor.predict(gait=parsed)
    return build_response(predictor, result)


@app.post("/predict/multimodal")
async def predict_multimodal(
    voice: UploadFile | None = File(None),
    handwriting: UploadFile | None = File(None),
    gait: UploadFile | None = File(None),
):
    if voice is None and handwriting is None and gait is None:
        raise HTTPException(400, "at least one modality file is required "
                                 "(voice, handwriting or gait)")
    predictor = get_predictor()
    kwargs: dict = {}
    if voice is not None:
        kwargs["voice"] = parse_voice_csv(await voice.read())
    if handwriting is not None:
        kwargs["handwriting"] = parse_handwriting_csv(await handwriting.read())
    if gait is not None:
        kwargs["gait"] = parse_gait_file(await gait.read(), predictor._feature_names["gait"])
    result = predictor.predict(**kwargs)
    return build_response(predictor, result)
