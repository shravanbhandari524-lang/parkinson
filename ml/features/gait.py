"""Feature engineering for PhysioNet VGRF gait time series.

Every function is reusable and operates on plain numpy arrays so it can be
unit-tested independently of the file layout. ``featurize_record`` turns one
raw record (``(n, 19)`` array as produced by
:func:`ml.data.loaders.load_gait_record`) into a flat feature vector;
``featurize_gait_records`` applies it to a whole dataset.

Conventions
-----------
* fs — sampling rate in Hz (inferred per record from the time column)
* contact — total vertical ground reaction force (VGRF) under one foot above
  ``CONTACT_THRESHOLD_N`` (the foot is in stance)
* onset — first sample of a contact episode (heel strike proxy)
* stride — consecutive onsets of the *same* foot (one full gait cycle)
* timing features use only *complete* contact episodes (records start and
  end mid-stride, so boundary-truncated contacts are excluded)

Degenerate records (too short, too few steps) yield ``NaN`` for the affected
features; the model pipelines impute them from training-fold medians.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from scipy.signal import welch

from ml.config import Config, get_config
from ml.data.loaders import GAIT_COLUMNS, Dataset, load_gait, load_gait_record

FS_DEFAULT = 100.0
CONTACT_THRESHOLD_N = 20.0  # ~2.5% of body weight: foot is on the ground
FMIN_HZ = 0.1
FMAX_HZ = 10.0
LOW_BAND_HZ = 1.0


# ---------------------------------------------------------------------------
# low-level reusable functions
# ---------------------------------------------------------------------------
def contact_mask(total_force: np.ndarray, threshold_n: float = CONTACT_THRESHOLD_N) -> np.ndarray:
    """Boolean mask of samples where a foot is in stance."""
    force = np.asarray(total_force, dtype=float)
    if force.ndim != 1:
        raise ValueError("total_force must be a 1-D signal")
    return force > threshold_n


def contact_episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` runs of a boolean contact mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def complete_contact_episodes(
    total_force: np.ndarray,
    threshold_n: float = CONTACT_THRESHOLD_N,
) -> list[tuple[int, int]]:
    """Contact episodes fully inside the record.

    Records typically start and end mid-stride, so the first/last contact is
    truncated; boundary-truncated episodes are excluded from *timing*
    features (stride/stance/swing/cadence) because their true onsets or ends
    lie outside the observation window.
    """
    episodes = contact_episodes(contact_mask(total_force, threshold_n))
    n = len(np.asarray(total_force))
    return [(s, e) for s, e in episodes if s > 0 and e < n]


def cadence_steps_per_min(
    left_total: np.ndarray,
    right_total: np.ndarray,
    fs: float,
    threshold_n: float = CONTACT_THRESHOLD_N,
) -> float:
    """Steps per minute over complete stance episodes of either foot."""
    n_steps = len(complete_contact_episodes(left_total, threshold_n)) + len(
        complete_contact_episodes(right_total, threshold_n)
    )
    duration_s = len(left_total) / fs
    if duration_s <= 0:
        return np.nan
    return float(n_steps / duration_s * 60.0)


def stride_intervals_s(
    total_force: np.ndarray,
    fs: float,
    threshold_n: float = CONTACT_THRESHOLD_N,
) -> np.ndarray:
    """Intervals (s) between successive same-foot onsets = stride times.

    Only complete contact episodes are used, so every interval is a fully
    observed gait cycle.
    """
    episodes = complete_contact_episodes(total_force, threshold_n)
    onsets = np.array([start for start, _ in episodes], dtype=float)
    if onsets.size < 3:
        return np.array([])
    return np.diff(onsets) / fs


def stride_time_stats(
    left_total: np.ndarray,
    right_total: np.ndarray,
    fs: float,
    threshold_n: float = CONTACT_THRESHOLD_N,
) -> dict[str, float]:
    """Stride time mean/std/CV pooled over both feet, plus per-foot means."""
    left = stride_intervals_s(left_total, fs, threshold_n)
    right = stride_intervals_s(right_total, fs, threshold_n)
    pooled = np.concatenate([left, right]) if (left.size + right.size) else np.array([])
    if pooled.size == 0:
        return {
            "stride_time_mean_s": np.nan,
            "stride_time_std_s": np.nan,
            "stride_time_cv": np.nan,
            "stride_time_mean_left_s": np.nan,
            "stride_time_mean_right_s": np.nan,
        }
    mean = float(np.mean(pooled))
    std = float(np.std(pooled, ddof=1)) if pooled.size > 1 else np.nan
    return {
        "stride_time_mean_s": mean,
        "stride_time_std_s": std,
        "stride_time_cv": float(std / mean) if np.isfinite(std) and mean > 0 else np.nan,
        "stride_time_mean_left_s": float(np.mean(left)) if left.size else np.nan,
        "stride_time_mean_right_s": float(np.mean(right)) if right.size else np.nan,
    }


def stance_swing_stats(
    left_total: np.ndarray,
    right_total: np.ndarray,
    fs: float,
    threshold_n: float = CONTACT_THRESHOLD_N,
) -> dict[str, float]:
    """Mean stance/swing durations, stance percentage and left/right symmetry."""
    stance_l = np.array([end - start for start, end in complete_contact_episodes(
        left_total, threshold_n)], dtype=float)
    stance_r = np.array([end - start for start, end in complete_contact_episodes(
        right_total, threshold_n)], dtype=float)
    stance_l, stance_r = stance_l / fs, stance_r / fs

    mean_stance_l = float(np.mean(stance_l)) if stance_l.size else np.nan
    mean_stance_r = float(np.mean(stance_r)) if stance_r.size else np.nan
    if not np.isfinite(mean_stance_l) and not np.isfinite(mean_stance_r):
        return {k: np.nan for k in (
            "stance_time_mean_s", "swing_time_mean_s", "stance_swing_ratio",
            "stance_pct", "lr_stance_symmetry")}

    ref = np.nanmean([mean_stance_l, mean_stance_r])
    mean_stride = None
    strides_l = stride_intervals_s(left_total, fs, threshold_n)
    strides_r = stride_intervals_s(right_total, fs, threshold_n)
    all_strides = np.concatenate([s for s in (strides_l, strides_r) if s.size])
    if all_strides.size:
        mean_stride = float(np.mean(all_strides))

    stance_time_mean = float(np.mean(np.concatenate(
        [s for s in (stance_l, stance_r) if s.size]))) if (stance_l.size + stance_r.size) else np.nan
    swing_time_mean = (mean_stride - stance_time_mean) if mean_stride else np.nan
    stance_pct = (stance_time_mean / mean_stride * 100.0) if mean_stride else np.nan
    stance_swing_ratio = (stance_time_mean / swing_time_mean) if (
        np.isfinite(swing_time_mean) and swing_time_mean > 0) else np.nan

    if np.isfinite(mean_stance_l) and np.isfinite(mean_stance_r):
        denom = 0.5 * (mean_stance_l + mean_stance_r)
        symmetry = (mean_stance_l - mean_stance_r) / denom if denom > 0 else np.nan
    else:
        symmetry = np.nan

    return {
        "stance_time_mean_s": stance_time_mean,
        "swing_time_mean_s": float(swing_time_mean) if np.isfinite(swing_time_mean) else np.nan,
        "stance_swing_ratio": float(stance_swing_ratio) if np.isfinite(stance_swing_ratio) else np.nan,
        "stance_pct": float(stance_pct) if np.isfinite(stance_pct) else np.nan,
        "lr_stance_symmetry": float(symmetry) if np.isfinite(symmetry) else np.nan,
    }


def force_stats(left_total: np.ndarray, right_total: np.ndarray) -> dict[str, float]:
    """Force level statistics and left/right asymmetry for the two feet."""
    left = np.asarray(left_total, dtype=float)
    right = np.asarray(right_total, dtype=float)
    combined = left + right

    def _cv(signal: np.ndarray) -> float:
        mean = float(np.mean(signal))
        return float(np.std(signal) / mean) if mean > 0 else np.nan

    mean_l, mean_r = float(np.mean(left)), float(np.mean(right))
    mean_c = float(np.mean(combined))
    asymmetry = (mean_l - mean_r) / mean_c if mean_c > 0 else np.nan
    return {
        "force_mean_N": mean_c,
        "force_max_N": float(np.max(combined)),
        "force_std_N": float(np.std(combined)),
        "force_cv": _cv(combined),
        "force_mean_left_N": mean_l,
        "force_mean_right_N": mean_r,
        "force_max_left_N": float(np.max(left)),
        "force_max_right_N": float(np.max(right)),
        "force_asymmetry": float(asymmetry) if np.isfinite(asymmetry) else np.nan,
        "abs_force_asymmetry": float(abs(asymmetry)) if np.isfinite(asymmetry) else np.nan,
    }


def loading_rate_stats(left_total: np.ndarray, right_total: np.ndarray, fs: float) -> dict[str, float]:
    """Vertical loading rate (N/s): PD gait shows reduced, gentler loading."""
    combined = np.asarray(left_total, dtype=float) + np.asarray(right_total, dtype=float)
    gradient = np.gradient(combined) * fs
    rising = gradient[gradient > 0]
    return {
        "loading_rate_mean_N_per_s": float(np.mean(rising)) if rising.size else np.nan,
        "loading_rate_p90_N_per_s": float(np.percentile(rising, 90)) if rising.size else np.nan,
    }


def power_spectrum(
    signal: np.ndarray,
    fs: float,
    fmin_hz: float = FMIN_HZ,
    fmax_hz: float = FMAX_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density restricted to ``[fmin_hz, fmax_hz]``."""
    signal = np.asarray(signal, dtype=float)
    signal = signal - np.nanmean(signal)
    nperseg = min(signal.size, int(4 * fs))  # ~4 s windows at 100 Hz
    freqs, psd = welch(signal, fs=fs, nperseg=max(nperseg, 16), detrend="constant")
    band = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    return freqs[band], psd[band]


def dominant_frequency(signal: np.ndarray, fs: float) -> float:
    """Frequency (Hz) of maximal power in the gait band."""
    freqs, psd = power_spectrum(signal, fs)
    if freqs.size == 0 or not np.any(psd > 0):
        return np.nan
    return float(freqs[int(np.argmax(psd))])


def spectral_entropy(signal: np.ndarray, fs: float) -> float:
    """Normalized Shannon entropy of the band-limited power spectrum in [0, 1].

    0 = single pure frequency (most regular), 1 = flat spectrum (most irregular).
    """
    freqs, psd = power_spectrum(signal, fs)
    total = float(np.sum(psd))
    if freqs.size < 2 or total <= 0:
        return np.nan
    p = psd / total
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(p.size))


def spectral_centroid(signal: np.ndarray, fs: float) -> float:
    """Power-weighted mean frequency (Hz) of the band-limited spectrum."""
    freqs, psd = power_spectrum(signal, fs)
    total = float(np.sum(psd))
    if freqs.size == 0 or total <= 0:
        return np.nan
    return float(np.sum(freqs * psd) / total)


def low_frequency_power_ratio(signal: np.ndarray, fs: float) -> float:
    """Fraction of band power below ``LOW_BAND_HZ`` (postural sway content)."""
    freqs, psd = power_spectrum(signal, fs)
    total = float(np.sum(psd))
    if total <= 0 or freqs.size == 0:
        return np.nan
    low = float(np.sum(psd[freqs < LOW_BAND_HZ]))
    return low / total


def signal_statistics(signal: np.ndarray) -> dict[str, float]:
    """Generic robust statistics of a 1-D signal."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return {k: np.nan for k in (
            "signal_skew", "signal_kurtosis", "signal_rms_N", "signal_peak_to_peak_N")}
    centered = signal - np.mean(signal)
    std = float(np.std(signal))
    skew = float(np.mean(centered**3) / std**3) if std > 0 else np.nan
    kurt = float(np.mean(centered**4) / std**4) if std > 0 else np.nan
    return {
        "signal_skew": skew,
        "signal_kurtosis": kurt,
        "signal_rms_N": float(np.sqrt(np.mean(signal**2))),
        "signal_peak_to_peak_N": float(np.max(signal) - np.min(signal)),
    }


def step_count(left_total: np.ndarray, right_total: np.ndarray,
               threshold_n: float = CONTACT_THRESHOLD_N) -> int:
    """Total complete stance episodes across both feet (step proxy)."""
    return len(complete_contact_episodes(left_total, threshold_n)) + len(
        complete_contact_episodes(right_total, threshold_n))


# ---------------------------------------------------------------------------
# per-record featurization
# ---------------------------------------------------------------------------
def infer_sampling_rate(record: np.ndarray) -> float:
    """Sampling rate from the record's time column (falls back to 100 Hz)."""
    time = record[:, 0]
    dt = np.diff(time)
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if dt.size else FS_DEFAULT


def featurize_record(record: np.ndarray, fs: float | None = None) -> dict[str, float]:
    """Turn one ``(n, 19)`` record into a flat ordered feature dict."""
    record = np.atleast_2d(np.asarray(record, dtype=float))
    if record.shape[1] != len(GAIT_COLUMNS):
        raise ValueError(f"expected {len(GAIT_COLUMNS)} columns, got {record.shape[1]}")
    fs = fs if fs is not None else infer_sampling_rate(record)

    left = record[:, 1:9].sum(axis=1)    # 8 left sensors
    right = record[:, 9:17].sum(axis=1)  # 8 right sensors
    duration_s = float(record[-1, 0] - record[0, 0]) if record.shape[0] > 1 else np.nan
    combined = left + right

    features: dict[str, float] = {
        "duration_s": duration_s,
        "n_steps": float(step_count(left, right)),
        "cadence_steps_per_min": cadence_steps_per_min(left, right, fs),
        "sampling_rate_hz": fs,
    }
    features.update(stride_time_stats(left, right, fs))
    features.update(stance_swing_stats(left, right, fs))
    features.update(force_stats(left, right))
    features.update(loading_rate_stats(left, right, fs))
    features.update({
        "dominant_frequency_hz": dominant_frequency(combined, fs),
        "spectral_entropy": spectral_entropy(combined, fs),
        "spectral_centroid_hz": spectral_centroid(combined, fs),
        "low_freq_power_ratio": low_frequency_power_ratio(combined, fs),
    })
    features.update(signal_statistics(combined))
    return features


def gait_feature_names() -> list[str]:
    """Ordered feature names produced by :func:`featurize_record`."""
    probe = np.zeros((200, len(GAIT_COLUMNS)), dtype=float)
    probe[:, 0] = np.arange(200) / FS_DEFAULT
    return list(featurize_record(probe, fs=FS_DEFAULT).keys())


def featurize_gait_records(
    source: Dataset | pd.DataFrame | Config | None = None,
    cfg: Config | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Featurize every gait record file and return a table indexed by record_id.

    ``source`` may be the gait :class:`~ml.data.loaders.Dataset`, a record
    table with ``record_id``/``path`` columns, or a Config (in which case the
    gait dataset is loaded from configuration).  The result merges 1:1 with
    the loader's descriptor table on ``record_id``.
    """
    if isinstance(source, Dataset):
        table = source.X
    elif isinstance(source, pd.DataFrame):
        table = source
    else:
        gait_ds = load_gait(cfg if cfg is not None else (source if isinstance(source, Config) else None))
        table = gait_ds.X

    required = {"record_id", "path"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"record table is missing columns: {sorted(missing)}")

    rows: dict[str, dict[str, float]] = {}
    total = len(table)
    for i, row in enumerate(table.itertuples(index=False), start=1):
        record_id = str(getattr(row, "record_id"))
        record = load_gait_record(getattr(row, "path"))
        rows[record_id] = featurize_record(record)
        if show_progress and i % 100 == 0:
            print(f"  featurized {i}/{total} gait records")

    features = pd.DataFrame.from_dict(rows, orient="index")
    features.index.name = "record_id"
    return features


def merge_gait_features(
    gait_dataset: Dataset,
    features: pd.DataFrame | Mapping[str, Mapping[str, float]] | None = None,
    cfg: Config | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Convenience: featurize (if needed) and return ``(X, y, groups)``.

    ``y``/``groups`` are aligned to ``X``'s record order.
    """
    if features is None:
        features = featurize_gait_records(gait_dataset)
    if not isinstance(features, pd.DataFrame):
        features = pd.DataFrame.from_dict(dict(features), orient="index")
    X = features.reindex(gait_dataset.X["record_id"].astype(str).values)
    return X, gait_dataset.y.reset_index(drop=True), gait_dataset.groups.reset_index(drop=True)


__all__ = [
    "FS_DEFAULT", "CONTACT_THRESHOLD_N", "FMIN_HZ", "FMAX_HZ",
    "contact_mask", "contact_episodes", "complete_contact_episodes",
    "cadence_steps_per_min",
    "stride_intervals_s", "stride_time_stats", "stance_swing_stats",
    "force_stats", "loading_rate_stats", "dominant_frequency",
    "spectral_entropy", "spectral_centroid", "low_frequency_power_ratio",
    "signal_statistics", "step_count", "infer_sampling_rate",
    "featurize_record", "gait_feature_names", "featurize_gait_records",
    "merge_gait_features",
]
