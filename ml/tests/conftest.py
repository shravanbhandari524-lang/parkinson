"""Shared pytest fixtures (session-scoped; loaders hit the real dataset files)."""

from __future__ import annotations

import pytest

from ml.config import Config, get_config
from ml.data.loaders import Dataset, load_gait, load_gait_record, load_handwriting, load_voice
from ml.features.gait import featurize_record


@pytest.fixture(scope="session")
def config() -> Config:
    return get_config()


@pytest.fixture(scope="session")
def voice_dataset(config) -> Dataset:
    return load_voice(config)


@pytest.fixture(scope="session")
def handwriting_dataset(config) -> Dataset:
    return load_handwriting(config)


@pytest.fixture(scope="session")
def gait_dataset(config) -> Dataset:
    return load_gait(config)


@pytest.fixture(scope="session")
def gait_record(config):
    """One real VGRF record (12,119 samples x 19 columns at 100 Hz)."""
    path = config.gait_path / "GaPt03_01.txt"
    return load_gait_record(path)


@pytest.fixture(scope="session")
def gait_sample_features(gait_record) -> dict[str, float]:
    return featurize_record(gait_record)
