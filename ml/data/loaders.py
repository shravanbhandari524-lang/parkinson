"""Dataset loaders with harmonized labels.

All three modalities are normalized to:

* ``y``: 1 = Parkinson's disease, 0 = healthy
* ``groups``: a patient/subject identifier that MUST be used as the grouping
  variable for any split of handwriting and gait data (also available for
  voice, see the note in :func:`load_voice`)
* ``metadata``: passthrough columns that are *not* features (identifiers,
  demographics, protocol descriptors)

No synthetic data is produced anywhere in this module; the loaders read the
real files referenced by ``ml/config.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ml.config import Config, get_config

PD_LABEL = 1  # Parkinson's disease
HEALTHY_LABEL = 0


# ---------------------------------------------------------------------------
# Shared container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    """A loaded modality: features, labels, grouping ids and passthrough metadata."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    metadata: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.feature_names:
            object.__setattr__(self, "feature_names", list(self.X.columns))
        n = len(self.X)
        if not (len(self.y) == len(self.groups) == len(self.metadata) == n):
            raise ValueError(
                f"{self.name}: X/y/groups/metadata length mismatch: "
                f"{n}/{len(self.y)}/{len(self.groups)}/{len(self.metadata)}"
            )

    @property
    def class_counts(self) -> dict[int, int]:
        return {int(k): int(v) for k, v in self.y.value_counts().sort_index().items()}

    @property
    def n_groups(self) -> int:
        return int(self.groups.nunique())


# ---------------------------------------------------------------------------
# Voice — UCI Parkinson's dataset (parkinsons.data)
# ---------------------------------------------------------------------------
def load_voice(cfg: Config | None = None) -> Dataset:
    """Load the UCI Parkinson's voice dataset.

    195 recordings x 22 acoustic features; ``status`` is already encoded as
    1 = PD / 0 = healthy, so labels are used verbatim.

    Grouping note
    -------------
    The ``name`` column encodes ``<cohort>_<registration>_<subject>_<rec>``
    (e.g. ``phon_R01_S01_1`` -> subject ``S01``).  The project spec mandates
    StratifiedKFold for this modality (recording-level), which is what the
    training code does by default; the parsed subject is still exposed in
    ``groups``/``metadata`` so a subject-level split can be enabled later.
    """
    cfg = cfg or get_config()
    path = cfg.voice_path
    if not path.exists():
        raise FileNotFoundError(f"voice dataset not found: {path}")

    df = pd.read_csv(path)
    required = {"name", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"voice dataset {path} is missing columns: {sorted(missing)}")

    y = df["status"].astype(int)
    unknown = sorted(set(y.unique()) - {HEALTHY_LABEL, PD_LABEL})
    if unknown:
        raise ValueError(f"voice labels must be 0/1, found: {unknown}")

    subject = df["name"].astype(str).str.extract(r"_([A-Za-z]+\d+)_\d+$")[0]
    metadata = pd.DataFrame(
        {
            "recording_id": df["name"].astype(str),
            "voice_subject": subject,
        }
    )
    X = df.drop(columns=["name", "status"])
    X = X.apply(pd.to_numeric, errors="raise")

    return Dataset(
        name="voice",
        X=X,
        y=y.rename("label"),
        groups=subject.rename("group"),
        metadata=metadata,
        description="UCI Parkinson's voice dataset (recording-level)",
    )


# ---------------------------------------------------------------------------
# Handwriting — HandPD (Spiral / Meander)
# ---------------------------------------------------------------------------
#: column renames that unify the spiral (ET suffix) and meander (ST suffix) variants
HANDWRITING_COLUMN_RENAMES: dict[str, str] = {
    "MAX_BETWEEN_ET_HT": "MAX_BETWEEN_STROKE_HT",
    "MIN_BETWEEN_ET_HT": "MIN_BETWEEN_STROKE_HT",
    "STD_DEVIATION_ET_HT": "STD_DEVIATION_STROKE_HT",
    "MAX_BETWEEN_ST_HT": "MAX_BETWEEN_STROKE_HT",
    "MIN_BETWEEN_ST_HT": "MIN_BETWEEN_STROKE_HT",
    "STD_DEVIATION_ST_HT": "STD_DEVIATION_STROKE_HT",
    "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ET_HT":
        "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_STROKE_HT",
    "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_ST_HT":
        "CHANGES_FROM_NEGATIVE_TO_POSITIVE_BETWEEN_STROKE_HT",
}

HANDWRITING_META_COLUMNS = ["_ID_EXAM", "IMAGE_NAME", "GENDER", "RIGH/LEFT-HANDED", "AGE"]


def _load_handpd_file(path: Path, exam_type: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=HANDWRITING_COLUMN_RENAMES)
    df = df.rename(columns={"ID_PATIENT": "patient_id", "CLASS_TYPE": "class_type"})
    required = {"patient_id", "class_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df.insert(0, "exam_type", exam_type)
    return df


def load_handwriting(cfg: Config | None = None) -> Dataset:
    """Load the HandPD handwriting dataset (spiral + meander exams).

    ``CLASS_TYPE`` 1 (healthy) / 2 (PD) is mapped to 0 / 1.  The grouping
    variable is ``(CLASS_TYPE, ID_PATIENT)`` — important data insight: the
    HandPD ``ID_PATIENT`` numbering restarts per class (healthy ids 1-18, PD
    ids 1-37 in these files), so patient "1" healthy and patient "1" PD are
    *different people*.  Spiral and meander use identical person numbering,
    so the combined group id keeps every person's exams together while
    keeping the two same-numbered people apart.
    """
    cfg = cfg or get_config()
    frames = [
        _load_handpd_file(cfg.handwriting_spiral_path, "spiral"),
        _load_handpd_file(cfg.handwriting_meander_path, "meander"),
    ]
    df = pd.concat(frames, ignore_index=True)

    y = (df["class_type"].astype(int) - 1).rename("label")
    unknown = sorted(set(y.unique()) - {HEALTHY_LABEL, PD_LABEL})
    if unknown:
        raise ValueError(f"CLASS_TYPE must map to 0/1, found: {unknown}")

    class_code = df["class_type"].map({1: "CO", 2: "PD"})
    groups = (
        "P" + df["patient_id"].astype(int).astype(str).str.zfill(2) + "_" + class_code
    ).rename("group")
    metadata = df[["exam_type"] + [c for c in HANDWRITING_META_COLUMNS if c in df.columns]].copy()
    metadata["patient_id"] = df["patient_id"].astype(int).values

    feature_cols = [c for c in df.columns if c not in {"exam_type", "class_type", "patient_id",
                                                       *HANDWRITING_META_COLUMNS}]
    X = df[feature_cols].apply(pd.to_numeric, errors="raise")

    return Dataset(
        name="handwriting",
        X=X,
        y=y,
        groups=groups,
        metadata=metadata,
        description="HandPD spiral + meander exams (patient-level grouping by ID_PATIENT)",
    )


# ---------------------------------------------------------------------------
# Gait — PhysioNet "Gait in Parkinson's Disease" (VGRF time series)
# ---------------------------------------------------------------------------
GAIT_RECORD_RE = re.compile(r"^(?P<task>Ga|Si|Ju)(?P<code>Co|Pt)(?P<num>\d+)_(?P<trial>\d+)\.txt$")

GAIT_COLUMNS: list[str] = (
    ["time_s"]
    + [f"left_sensor_{i}" for i in range(1, 9)]
    + [f"right_sensor_{i}" for i in range(1, 9)]
    + ["total_force_left_N", "total_force_right_N"]
)


def list_gait_records(root: str | Path) -> pd.DataFrame:
    """Index all PhysioNet gait record files.

    Returns a table with one row per record file: record id, subject/person
    group, label (Pt = PD -> 1, Co = control -> 0), protocol task (Ga/Si/Ju),
    trial number and absolute path.

    Grouping note: the Ga/Si/Ju prefixes denote three *disjoint* cohorts
    (verified against demographics.txt: e.g. GaPt03 is 82 years old while
    JuPt03 is 74), so the full task-prefixed id (``GaPt03``) is the unique
    person identifier and the grouping variable.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"gait dataset directory not found: {root}")

    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*.txt")):
        match = GAIT_RECORD_RE.match(path.name)
        if match is None:
            continue  # demographics.txt, format.txt, SHA256SUMS.txt, ...
        subject = f"{match['task']}{match['code']}{int(match['num']):02d}"  # unique person id
        rows.append(
            {
                "record_id": path.stem,
                "path": str(path),
                "subject": subject,
                "person_code": match["code"],
                "label": PD_LABEL if match["code"] == "Pt" else HEALTHY_LABEL,
                "task": match["task"],
                "trial": int(match["trial"]),
            }
        )
    if not rows:
        raise ValueError(f"no gait record files matching the PhysioNet pattern in {root}")
    return pd.DataFrame(rows)


def load_gait_record(path: str | Path) -> np.ndarray:
    """Read one record file into an ``(n_samples, 19)`` float array.

    Columns follow ``GAIT_COLUMNS``: time, 8 left VGRF sensors, 8 right VGRF
    sensors, total left force, total right force (tab-separated, 100 Hz).
    """
    data = np.loadtxt(path, delimiter="\t", comments=None, encoding="utf-8")
    data = np.atleast_2d(data)
    if data.shape[1] != len(GAIT_COLUMNS):
        raise ValueError(f"{path}: expected {len(GAIT_COLUMNS)} columns, got {data.shape[1]}")
    return data.astype(float)


def _sampling_rate(record: np.ndarray) -> float:
    """Nominal sampling rate from the time column (robust to irregular steps)."""
    dt = np.diff(record[:, 0])
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("time column is non-increasing; cannot infer sampling rate")
    return float(1.0 / np.median(dt))


def _load_demographics(root: Path) -> pd.DataFrame | None:
    """Best-effort parse of demographics.txt (metadata only; never fatal).

    Each demographics row is keyed by the full record prefix (``GaPt03``),
    which is exactly the subject/person id used for grouping.
    """
    path = root / "demographics.txt"
    if not path.exists():
        return None
    try:
        demo = pd.read_csv(path, sep=r"\s+", engine="python")
        demo = demo.loc[:, ~demo.columns.str.startswith("Unnamed")]
        demo = demo.dropna(axis=1, how="all")
        if "ID" not in demo.columns:
            return None
        demo["subject"] = demo["ID"].astype(str).str.strip()
        demo = demo.drop_duplicates(subset="subject")
        return demo
    except Exception:  # noqa: BLE001 - demographics are optional metadata
        return None


def load_gait(cfg: Config | None = None) -> Dataset:
    """Load the PhysioNet gait dataset as a record-level table.

    The returned ``X`` is a *descriptor* table (one row per record file) — the
    actual VGRF features are engineered per record by
    :func:`ml.features.gait.featurize_gait_records` and merged on
    ``record_id``.  ``groups`` is the unique subject/person id (e.g.
    ``GaPt03``; the Ga/Si/Ju task cohorts are disjoint people), which makes
    subject-level splitting mandatory and sufficient.
    """
    cfg = cfg or get_config()
    root = cfg.gait_path
    records = list_gait_records(root)

    if "path" not in records.columns:  # pragma: no cover - defensive
        raise ValueError("record table is missing the path column")

    demo = _load_demographics(root)
    if demo is not None:
        keep = [c for c in ("subject", "Age", "Height", "Weight", "HoehnYahr", "UPDRS", "TUAG")
                if c in demo.columns]
        records = records.merge(demo[keep], on="subject", how="left")

    counts = records["label"].value_counts().sort_index()
    if set(counts.index) != {HEALTHY_LABEL, PD_LABEL}:
        raise ValueError(f"gait labels must include both classes, found: {counts.to_dict()}")

    descriptor = records.drop(columns=[c for c in ("person_code",) if c in records.columns])
    metadata_cols = [c for c in descriptor.columns
                     if c not in {"record_id", "path", "subject", "label"}]
    metadata = descriptor[["record_id"] + metadata_cols].set_index("record_id")

    return Dataset(
        name="gait",
        X=descriptor[["record_id", "path"]],
        y=records["label"].rename("label"),
        groups=records["subject"].rename("group"),
        metadata=metadata.reset_index(drop=True),
        description="PhysioNet Gait in Parkinson's Disease (subject-level grouping)",
    )


def iter_gait_files(root: str | Path) -> Iterable[tuple[str, Path]]:
    """Yield ``(record_id, path)`` for every gait record file."""
    for _, row in list_gait_records(root).iterrows():
        yield str(row["record_id"]), Path(row["path"])
