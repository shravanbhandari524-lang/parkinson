"""Gait feature-extraction tests.

Synthetic signals here are unit-test fixtures with analytically known
properties (cadence, stride timing, frequency); they are never used for
training — all training uses the real record files.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.features.gait import (
    contact_episodes,
    contact_mask,
    cadence_steps_per_min,
    dominant_frequency,
    featurize_record,
    force_stats,
    gait_feature_names,
    spectral_entropy,
    stance_swing_stats,
    stride_time_stats,
)


def _contact_train(fs: float, duration_s: float, period_s: float, duty: float,
                   offset_s: float = 0.0, peak_n: float = 400.0) -> np.ndarray:
    """Rectangular stance pulses for one foot (no wrap-around artifacts)."""
    t = np.arange(int(duration_s * fs)) / fs
    signal = np.zeros_like(t)
    start = offset_s
    while start < duration_s:
        end = start + duty * period_s
        signal[(t >= start) & (t < min(end, duration_s))] = peak_n
        start += period_s
    return signal


def test_contact_mask_and_episodes():
    mask = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    force = np.array([0, 5, 6, 7, 0, 0, 3, 3, 0], dtype=float)
    assert contact_mask(force, threshold_n=2).tolist() == mask.tolist()
    episodes = contact_episodes(mask)
    assert episodes == [(1, 4), (6, 8)]


def test_cadence_known_pattern():
    fs, duration = 100.0, 21.5
    # each foot steps every 0.5 s -> 2 steps/s per foot -> 240 steps/min total
    left = _contact_train(fs, duration, period_s=0.5, duty=0.4)
    right = _contact_train(fs, duration, period_s=0.5, duty=0.4, offset_s=0.25)
    cadence = cadence_steps_per_min(left, right, fs)
    assert cadence == pytest.approx(240.0, rel=0.02)


def test_stride_time_known_pattern():
    fs, duration = 100.0, 21.5
    left = _contact_train(fs, duration, period_s=1.0, duty=0.6)   # stride = 1.0 s
    right = _contact_train(fs, duration, period_s=1.0, duty=0.6, offset_s=0.5)
    stats = stride_time_stats(left, right, fs)
    assert stats["stride_time_mean_s"] == pytest.approx(1.0, rel=0.01)
    assert stats["stride_time_std_s"] == pytest.approx(0.0, abs=1e-6)
    assert stats["stride_time_cv"] == pytest.approx(0.0, abs=1e-6)
    assert stats["stride_time_mean_left_s"] == pytest.approx(1.0, rel=0.01)
    assert stats["stride_time_mean_right_s"] == pytest.approx(1.0, rel=0.01)


def test_stance_swing_known_pattern():
    fs, duration = 100.0, 21.5
    left = _contact_train(fs, duration, period_s=1.0, duty=0.6)
    right = _contact_train(fs, duration, period_s=1.0, duty=0.6, offset_s=0.5)
    stats = stance_swing_stats(left, right, fs)
    assert stats["stance_time_mean_s"] == pytest.approx(0.6, rel=0.01)
    assert stats["swing_time_mean_s"] == pytest.approx(0.4, rel=0.01)
    assert stats["stance_pct"] == pytest.approx(60.0, rel=0.01)
    assert stats["lr_stance_symmetry"] == pytest.approx(0.0, abs=1e-6)  # perfectly symmetric


def test_stance_symmetry_detects_asymmetry():
    fs, duration = 100.0, 21.5
    left = _contact_train(fs, duration, period_s=1.0, duty=0.7)   # longer stance left
    right = _contact_train(fs, duration, period_s=1.0, duty=0.5, offset_s=0.5)
    stats = stance_swing_stats(left, right, fs)
    assert stats["lr_stance_symmetry"] > 0.1  # signed: left stance longer than right


def test_force_stats_asymmetry_sign():
    rng = np.random.default_rng(0)
    left = 500.0 + rng.normal(0, 10, size=2000)
    right = 400.0 + rng.normal(0, 10, size=2000)
    stats = force_stats(left, right)
    assert stats["force_asymmetry"] == pytest.approx(100.0 / 900.0, rel=0.05)  # (L-R)/combined
    assert stats["force_asymmetry"] > 0
    assert stats["force_mean_N"] == pytest.approx(900.0, rel=0.01)
    assert stats["force_cv"] > 0


def test_dominant_frequency_of_sine():
    fs = 100.0
    t = np.arange(4000) / fs
    signal = 100.0 * np.sin(2 * np.pi * 2.0 * t)  # 2 Hz
    assert dominant_frequency(signal, fs) == pytest.approx(2.0, abs=0.1)


def test_spectral_entropy_orders_regularity():
    fs = 100.0
    rng = np.random.default_rng(42)
    t = np.arange(8000) / fs
    sine = np.sin(2 * np.pi * 1.8 * t)
    noise = rng.normal(0, 1, size=8000)
    assert spectral_entropy(sine, fs) < spectral_entropy(noise, fs)
    assert 0.0 < spectral_entropy(sine, fs) < 1.0


def test_featurize_real_record(gait_record, gait_sample_features):
    feats = gait_sample_features
    core = [
        "cadence_steps_per_min", "stride_time_mean_s", "stride_time_std_s",
        "stride_time_cv", "stance_time_mean_s", "swing_time_mean_s",
        "force_mean_N", "force_max_N", "force_cv", "force_asymmetry",
        "dominant_frequency_hz", "spectral_entropy", "loading_rate_mean_N_per_s",
    ]
    for name in core:
        assert np.isfinite(feats[name]), f"{name} is not finite for a healthy-length record"
    assert 60 < feats["cadence_steps_per_min"] < 200
    assert 0.6 < feats["stride_time_mean_s"] < 2.0
    assert feats["force_max_N"] > feats["force_mean_N"] > 0
    assert 0 <= feats["spectral_entropy"] <= 1
    assert 0.5 < feats["dominant_frequency_hz"] < 5.0


def test_feature_names_contract():
    names = gait_feature_names()
    assert len(names) == 34
    required = {
        "cadence_steps_per_min", "stride_time_mean_s", "stride_time_std_s",
        "stride_time_cv", "stance_time_mean_s", "swing_time_mean_s",
        "lr_stance_symmetry", "force_mean_N", "force_max_N", "force_std_N",
        "force_cv", "force_asymmetry", "dominant_frequency_hz", "spectral_entropy",
    }
    assert required.issubset(set(names))
    assert len(set(names)) == len(names)  # no duplicates


def test_featurize_rejects_wrong_shape():
    with pytest.raises(ValueError, match="expected 19 columns"):
        featurize_record(np.ones((100, 5)))
