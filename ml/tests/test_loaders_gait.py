"""Gait loader tests against the real PhysioNet record files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data.loaders import GAIT_COLUMNS, list_gait_records, load_gait_record


def test_gait_record_counts(gait_dataset):
    """306 record files: 214 PD, 92 control."""
    assert len(gait_dataset.X) == 306
    assert gait_dataset.class_counts == {0: 92, 1: 214}


def test_gait_subject_counts_and_grouping(gait_dataset):
    """93 PD + 72 control subjects with record files; groups are task-prefixed ids."""
    subjects = gait_dataset.groups
    n_pd = int(subjects[subjects.str.contains("Pt")].nunique())
    n_co = int(subjects[subjects.str.contains("Co")].nunique())
    assert (n_pd, n_co) == (93, 72)
    assert subjects.str.match(r"^(Ga|Si|Ju)(Co|Pt)\d{2}$").all()
    # e.g. GaPt03 — the Ga/Si/Ju cohorts are disjoint people, so the full id is the person
    assert set(subjects.str[:2]) == {"Ga", "Si", "Ju"}


def test_gait_multiple_records_per_subject(gait_dataset):
    sizes = gait_dataset.groups.value_counts()
    assert (sizes >= 1).all()
    assert sizes.max() >= 5  # e.g. JuPt01 has 7 trials
    assert int((sizes > 1).sum()) > 50  # many subjects have several trials


def test_gait_label_consistent_within_subject(gait_dataset):
    df = pd.DataFrame({"group": gait_dataset.groups, "y": gait_dataset.y})
    per_subject = df.groupby("group")["y"].nunique()
    assert (per_subject == 1).all()


def test_gait_metadata_not_duplicated_by_demographics(gait_dataset):
    """The demographics merge must not multiply rows."""
    md = gait_dataset.metadata
    assert len(md) == len(gait_dataset.X)
    assert "task" in md.columns and "trial" in md.columns


def test_gait_trial_metadata(gait_dataset):
    md = gait_dataset.metadata
    assert (md["trial"] >= 1).all()
    assert set(md["task"]) == {"Ga", "Si", "Ju"}


def test_gait_record_loading(config, gait_record):
    assert gait_record.shape[1] == len(GAIT_COLUMNS) == 19
    assert gait_record.shape[0] == 12119  # GaPt03_01: ~2 minutes at 100 Hz
    dt = np.diff(gait_record[:, 0])
    assert abs(1.0 / np.median(dt) - 100.0) < 0.5  # nominal 100 Hz
    assert np.isfinite(gait_record).all()


def test_gait_record_bad_column_count(tmp_path):
    bad = tmp_path / "bad.txt"
    np.savetxt(bad, np.ones((10, 5)), delimiter="\t")
    with pytest.raises(ValueError, match="expected 19 columns"):
        load_gait_record(bad)


def test_list_gait_records_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_gait_records(tmp_path / "does_not_exist")
