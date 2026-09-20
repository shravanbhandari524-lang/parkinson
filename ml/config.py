"""Configuration system.

Values are read from ``ml/config.yaml``; optional ``ml/config.local.yaml``
overrides them; environment variables (see ``ENV_OVERRIDES``) override both.
A ``.env`` file in the repository root is honored via python-dotenv.

Paths inside the config are stored relative to either the repository root
(dataset paths, relative to ``data_dir``) or the ``ml/`` package directory
(artifact/metric output dirs) and are resolved by :class:`Config`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

ML_PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_PACKAGE_DIR.parent
DEFAULT_CONFIG_PATH = ML_PACKAGE_DIR / "config.yaml"
LOCAL_CONFIG_PATH = ML_PACKAGE_DIR / "config.local.yaml"

#: environment variable -> dotted config path it overrides
ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "PARK_ML_DATA_DIR": ("data_dir",),
    "PARK_ML_VOICE_PATH": ("voice", "path"),
    "PARK_ML_HANDWRITING_SPIRAL_PATH": ("handwriting", "spiral_path"),
    "PARK_ML_HANDWRITING_MEANDER_PATH": ("handwriting", "meander_path"),
    "PARK_ML_GAIT_PATH": ("gait", "path"),
    "PARK_ML_ARTIFACTS_DIR": ("artifacts_dir",),
    "PARK_ML_METRICS_DIR": ("metrics_dir",),
    "PARK_ML_SEED": ("seed",),
}

INT_KEYS = {("seed",)}


def _deep_update(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``extra`` into ``base`` (in place) and return it."""
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_nested(cfg: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = cfg
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the merged configuration dictionary (files, then environment)."""
    load_dotenv(REPO_ROOT / ".env", override=False)

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}

    if not config_path and LOCAL_CONFIG_PATH.exists():
        with open(LOCAL_CONFIG_PATH, "r", encoding="utf-8") as fh:
            local = yaml.safe_load(fh) or {}
        _deep_update(cfg, local)

    for env_name, dotted in ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is not None:
            value: Any = int(raw) if dotted in INT_KEYS else raw
            _set_nested(cfg, dotted, value)

    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    for key in ("data_dir", "voice", "handwriting", "gait"):
        if key not in cfg:
            raise KeyError(f"config is missing required key: {key}")
    for section, attr in (("voice", "path"), ("handwriting", "spiral_path"),
                          ("handwriting", "meander_path"), ("gait", "path")):
        if attr not in cfg[section]:
            raise KeyError(f"config is missing required key: {section}.{attr}")
    seed = cfg.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ValueError("seed must be an integer")


class Config:
    """Read-only accessor over the merged configuration dictionary."""

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self._raw: dict[str, Any] = dict(raw)

    # -- generic access -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    @property
    def raw(self) -> dict[str, Any]:
        return dict(self._raw)

    def section(self, name: str) -> dict[str, Any]:
        return dict(self._raw.get(name, {}))

    # -- paths ----------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return (REPO_ROOT / str(self._raw["data_dir"])).resolve()

    @property
    def artifacts_dir(self) -> Path:
        value = str(self._raw.get("artifacts_dir", "artifacts"))
        path = Path(value)
        return (path if path.is_absolute() else ML_PACKAGE_DIR / path).resolve()

    @property
    def metrics_dir(self) -> Path:
        value = str(self._raw.get("metrics_dir", "metrics"))
        path = Path(value)
        return (path if path.is_absolute() else ML_PACKAGE_DIR / path).resolve()

    def _data_path(self, *parts: str) -> Path:
        return (self.data_dir / Path(*parts)).resolve()

    @property
    def voice_path(self) -> Path:
        return self._data_path(self._raw["voice"]["path"])

    @property
    def handwriting_spiral_path(self) -> Path:
        return self._data_path(self._raw["handwriting"]["spiral_path"])

    @property
    def handwriting_meander_path(self) -> Path:
        return self._data_path(self._raw["handwriting"]["meander_path"])

    @property
    def gait_path(self) -> Path:
        return self._data_path(self._raw["gait"]["path"])

    @property
    def seed(self) -> int:
        return int(self._raw.get("seed", 42))


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached configuration (honors env vars set before first call)."""
    return Config(load_config())


def clear_config_cache() -> None:
    """Reset the cached config (useful in tests after env changes)."""
    get_config.cache_clear()
