"""Internal ML HTTP service.

Exposes the trained multimodal PD models over HTTP for the Express.js
backend. This service is *internal only* — it is never exposed to the
internet; the Node/Express API is the public entry point.

Endpoints
---------
GET  /health                    liveness + artifact status
GET  /model-info                trained model metadata and CV metrics
POST /predict/voice             CSV upload -> voice PD probability
POST /predict/handwriting       CSV upload -> handwriting PD probability
POST /predict/gait              TXT upload -> gait PD probability
POST /predict/multimodal        any subset of the three uploads -> fused result

Response shape (prediction endpoints)
-------------------------------------
{
  "voice":       {"probability": 0.0, "top_features": [...]},   # only supplied modalities
  "handwriting": {...},
  "gait":        {...},
  "multimodal":  {"probability": 0.0, "top_features": [...]},
  "risk_band":   "low" | "borderline" | "high",
  "model_version": "1.0.0"
}

Probabilities always come from the trained artifacts — nothing is hard-coded.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from ml.config import get_config
from ml.data.loaders import HANDWRITING_COLUMN_RENAMES
from ml.features.gait import featurize_record
from ml.inference.predict import MultimodalPredictor

logger = logging.getLogger("ml.service")

MODEL_VERSION = "1.0.0"

#: (low, high) thresholds for the risk band, from the project plan
RISK_BAND_LOW = 0.35
RISK_BAND_HIGH = 0.65

#: gait VGRF records are (n_samples, 19) time/force tables
GAIT_N_COLUMNS = 19

METRICS = ("voice", "handwriting", "gait", "fusion")


def risk_band(probability: float) -> str:
    if probability < RISK_BAND_LOW:
        return "low"
    if probability <= RISK_BAND_HIGH:
        return "borderline"
    return "high"


def _load_metrics(cfg) -> dict[str, dict[str, Any]]:
    """Read the training metric reports written by ``ml/train_all.py``."""
    reports: dict[str, dict[str, Any]] = {}
    for name in METRICS:
        path = cfg.metrics_dir / f"{name}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run ml/train_all.py first")
        with open(path, encoding="utf-8") as fh:
            reports[name] = json.load(fh)
    return reports


def _require_predictor(request: Request) -> MultimodalPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        detail = getattr(request.app.state, "load_error", None) or \
            "ML models are not loaded"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ML service degraded: {detail}",
        )
    return predictor


# --------------------------------------------------------------------- inputs
def _voice_frame(content: bytes, feature_names: list[str]) -> pd.DataFrame:
    """Parse an uploaded voice CSV into one feature row (mean over rows)."""
    df = pd.read_csv(io.BytesIO(content))
    df = df.drop(columns=[c for c in ("name", "status") if c in df.columns])
    missing = [n for n in feature_names if n not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"voice CSV is missing required columns: {missing[:5]}",
        )
    if df.empty:
        raise HTTPException(status_code=422, detail="voice CSV contains no data rows")
    return df[list(feature_names)].mean(numeric_only=True).to_frame().T


def _handwriting_frame(content: bytes, feature_names: list[str]) -> pd.DataFrame:
    """Parse an uploaded HandPD CSV (spiral or meander) into one feature row."""
    df = pd.read_csv(io.BytesIO(content))
    # raw HandPD files use ET (spiral) / ST (meander) suffixes; training used
    # the canonical *_STROKE_HT names, so apply the same rename map
    df = df.rename(columns=HANDWRITING_COLUMN_RENAMES)
    missing = [n for n in feature_names if n not in df.columns]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"handwriting CSV is missing required columns: {missing[:5]}",
        )
    if df.empty:
        raise HTTPException(status_code=422, detail="handwriting CSV contains no data rows")
    return df[list(feature_names)].mean(numeric_only=True).to_frame().T


def _gait_array(content: bytes) -> np.ndarray:
    """Parse an uploaded PhysioNet VGRF record (tab/whitespace separated)."""
    try:
        array = np.loadtxt(io.BytesIO(content), ndmin=2)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"gait file is not a valid numeric table: {exc}",
        ) from exc
    if array.shape[1] != GAIT_N_COLUMNS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"gait record must have {GAIT_N_COLUMNS} columns, got {array.shape[1]}",
        )
    if array.shape[0] == 0:
        raise HTTPException(status_code=422, detail="gait file contains no samples")
    return array


# -------------------------------------------------------------------- output
def _modality_entry(result: Mapping[str, Any], modality: str) -> dict[str, Any]:
    return {
        "probability": float(result["per_modality_probability"][modality]),
        "top_features": [
            {k: v for k, v in item.items() if k != "abs_contribution"}
            for item in result["top_features"]
            if item["modality"] == modality
        ],
    }


def _build_response(result: Mapping[str, Any], modalities: list[str]) -> dict[str, Any]:
    """Shape a predictor result into the multimodal API contract."""
    response: dict[str, Any] = {m: _modality_entry(result, m) for m in modalities}
    response["multimodal"] = {
        "probability": float(result["probability"]),
        "top_features": result["top_features"],
    }
    response["risk_band"] = risk_band(float(result["probability"]))
    response["model_version"] = MODEL_VERSION
    return response


# ----------------------------------------------------------------- lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    app.state.cfg = cfg
    app.state.predictor = None
    app.state.load_error = None
    try:
        app.state.predictor = MultimodalPredictor(cfg)
        logger.info("ML models loaded from %s", cfg.artifacts_dir)
    except Exception as exc:  # degraded mode: service starts but cannot predict
        app.state.load_error = str(exc)
        logger.error("failed to load ML artifacts: %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Parkinson's Multimodal ML Service",
        description="Internal inference service for the Express.js backend.",
        version=MODEL_VERSION,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------ health/meta
    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        loaded = request.app.state.predictor is not None
        body = {
            "status": "ok" if loaded else "degraded",
            "model_version": MODEL_VERSION,
            "models_loaded": loaded,
        }
        if not loaded:
            body["detail"] = getattr(request.app.state, "load_error", None)
            return JSONResponse(body, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return JSONResponse(body)

    @app.get("/model-info")
    async def model_info(request: Request) -> JSONResponse:
        try:
            reports = _load_metrics(request.app.state.cfg)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        info: dict[str, Any] = {"model_version": MODEL_VERSION, "modalities": {}}
        for name in ("voice", "handwriting", "gait"):
            report = reports[name]
            primary = report["primary_model"]
            aggregate = report["models"][primary]["aggregate"]
            info["modalities"][name] = {
                "primary_model": primary,
                "split_strategy": report["split_strategy"],
                "n_samples": report["n_samples"],
                "n_features": report["n_features"],
                "metrics": {
                    "accuracy": aggregate.get("accuracy_mean"),
                    "sensitivity": aggregate.get("sensitivity_mean"),
                    "specificity": aggregate.get("specificity_mean"),
                    "roc_auc": aggregate.get("roc_auc_mean"),
                },
            }
        fusion = reports["fusion"]
        info["fusion"] = {
            "modalities": list(fusion.get("modalities", ["voice", "handwriting", "gait"])),
            "risk_bands": {
                "low": f"< {RISK_BAND_LOW}",
                "borderline": f"{RISK_BAND_LOW} - {RISK_BAND_HIGH}",
                "high": f"> {RISK_BAND_HIGH}",
            },
        }
        return JSONResponse(info)

    # ------------------------------------------------------------- prediction
    @app.post("/predict/voice")
    async def predict_voice(request: Request, file: UploadFile = File(...)):
        predictor = _require_predictor(request)
        content = await file.read()
        frame = _voice_frame(content, predictor._feature_names["voice"])
        result = predictor.predict(voice=frame)
        return JSONResponse(_build_response(result, ["voice"]))

    @app.post("/predict/handwriting")
    async def predict_handwriting(request: Request, file: UploadFile = File(...)):
        predictor = _require_predictor(request)
        content = await file.read()
        frame = _handwriting_frame(content, predictor._feature_names["handwriting"])
        result = predictor.predict(handwriting=frame)
        return JSONResponse(_build_response(result, ["handwriting"]))

    @app.post("/predict/gait")
    async def predict_gait(request: Request, file: UploadFile = File(...)):
        predictor = _require_predictor(request)
        content = await file.read()
        array = _gait_array(content)
        result = predictor.predict(gait=array)
        return JSONResponse(_build_response(result, ["gait"]))

    @app.post("/predict/multimodal")
    async def predict_multimodal(
        request: Request,
        voice: UploadFile | None = File(None),
        handwriting: UploadFile | None = File(None),
        gait: UploadFile | None = File(None),
    ):
        predictor = _require_predictor(request)
        provided: dict[str, Any] = {}
        if voice is not None and voice.filename:
            provided["voice"] = _voice_frame(
                await voice.read(), predictor._feature_names["voice"])
        if handwriting is not None and handwriting.filename:
            provided["handwriting"] = _handwriting_frame(
                await handwriting.read(), predictor._feature_names["handwriting"])
        if gait is not None and gait.filename:
            provided["gait"] = _gait_array(await gait.read())

        if not provided:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="provide at least one modality file: voice, handwriting or gait",
            )

        result = predictor.predict(**provided)
        return JSONResponse(_build_response(result, list(provided)))

    return app


app = create_app()
