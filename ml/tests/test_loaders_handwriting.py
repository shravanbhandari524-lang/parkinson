"""Handwriting loader tests against the real HandPD CSV files."""

from __future__ import annotations

import pandas as pd

from ml.data.loaders import HANDWRITING_COLUMN_RENAMES


def test_handwriting_shape(handwriting_dataset):
    # ~368 exam rows per file, spiral + meander combined
    assert len(handwriting_dataset.X) == 736
    assert handwriting_dataset.X.shape[1] == 9


def test_handwriting_label_mapping(handwriting_dataset):
    """CLASS_TYPE 1 (healthy) -> 0, CLASS_TYPE 2 (PD) -> 1."""
    y = handwriting_dataset.y
    assert set(y.unique()) == {0, 1}
    assert len(y) == 736


def test_handwriting_label_mapping_against_raw(handwriting_dataset, config):
    raw = pd.read_csv(config.handwriting_spiral_path)
    expected_pd = int((raw["CLASS_TYPE"] == 2).sum())
    expected_healthy = int((raw["CLASS_TYPE"] == 1).sum())
    spiral_y = handwriting_dataset.y[handwriting_dataset.metadata["exam_type"] == "spiral"]
    assert int((spiral_y == 1).sum()) == expected_pd
    assert int((spiral_y == 0).sum()) == expected_healthy


def test_handwriting_features_unified_names(handwriting_dataset):
    renamed = set(HANDWRITING_COLUMN_RENAMES.values())
    assert renamed.issubset(set(handwriting_dataset.feature_names))
    # no raw ET/ST suffixes survive the unification
    assert not any("_ET_" in c or "_ST_" in c for c in handwriting_dataset.feature_names)


def test_handwriting_groups_are_patients(handwriting_dataset):
    groups = handwriting_dataset.groups
    assert groups.notna().all()
    assert groups.str.match(r"^P\d{2}_(CO|PD)$").all()
    # same person performed both spiral and meander exams
    md = handwriting_dataset.metadata
    spiral_patients = set(groups[md["exam_type"] == "spiral"])
    meander_patients = set(groups[md["exam_type"] == "meander"])
    assert spiral_patients == meander_patients
    # patients contribute multiple rows -> group split is mandatory
    sizes = groups.value_counts()
    assert sizes.max() > 2


def test_handwriting_patient_ids_restart_per_class(handwriting_dataset):
    """HandPD patient numbering restarts per class: 18 healthy + 36 PD people
    (PD ids run 1-37 but id 4 is absent from these files)."""
    groups = handwriting_dataset.groups
    n_co = int(groups[groups.str.endswith("_CO")].nunique())
    n_pd = int(groups[groups.str.endswith("_PD")].nunique())
    assert (n_co, n_pd) == (18, 36)
    # ...so the raw id alone is ambiguous and the class must be part of the key
    assert "P01_CO" in set(groups) and "P01_PD" in set(groups)


def test_handwriting_label_consistent_within_patient(handwriting_dataset):
    """A patient's spiral and meander exams must carry the same diagnosis."""
    df = pd.DataFrame({"group": handwriting_dataset.groups, "y": handwriting_dataset.y})
    per_patient = df.groupby("group")["y"].nunique()
    assert (per_patient == 1).all()
