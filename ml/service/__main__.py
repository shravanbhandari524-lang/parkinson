"""Run the internal ML service: ``ml/.venv/bin/python -m ml.service``."""
from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "ml.service.app:app",
        host=os.environ.get("ML_SERVICE_HOST", "127.0.0.1"),
        port=int(os.environ.get("ML_SERVICE_PORT", "8000")),
        log_level=os.environ.get("ML_SERVICE_LOG_LEVEL", "info"),
    )
