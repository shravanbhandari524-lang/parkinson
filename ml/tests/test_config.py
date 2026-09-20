"""Configuration system tests (paths resolve, env vars override)."""

from __future__ import annotations

from pathlib import Path

from ml.config import ENV_OVERRIDES, REPO_ROOT, load_config


def test_default_config_resolves_existing_paths(config):
    assert config.voice_path.exists()
    assert config.handwriting_spiral_path.exists()
    assert config.handwriting_meander_path.exists()
    assert config.gait_path.is_dir()
    assert config.data_dir.exists()


def test_config_paths_live_under_repo_root(config):
    for path in (config.voice_path, config.handwriting_spiral_path, config.gait_path):
        assert REPO_ROOT in path.parents or path.parent == REPO_ROOT or REPO_ROOT in Path(path).parents


def test_seed_is_int_and_stable(config):
    assert isinstance(config.seed, int)


def test_env_overrides_take_precedence(monkeypatch, tmp_path):
    fake_voice = tmp_path / "voice.csv"
    fake_voice.write_text("name,status\nx,1\n")
    monkeypatch.setenv("PARK_ML_VOICE_PATH", str(fake_voice))

    cfg = load_config()
    assert cfg["voice"]["path"] == str(fake_voice)
    assert ENV_OVERRIDES["PARK_ML_VOICE_PATH"] == ("voice", "path")


def test_env_seed_override(monkeypatch):
    monkeypatch.setenv("PARK_ML_SEED", "7")
    cfg = load_config()
    assert cfg["seed"] == 7
