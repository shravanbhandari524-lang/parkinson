"""Voice loader tests against the real UCI parkinsons.data file."""

from __future__ import annotations

import pandas as pd


def test_voice_shape(voice_dataset):
    assert voice_dataset.X.shape == (195, 22)
    assert len(voice_dataset.y) == 195
    assert len(voice_dataset.groups) == 195


def test_voice_feature_names(voice_dataset):
    expected_head = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]
    assert voice_dataset.feature_names[:3] == expected_head
    assert len(voice_dataset.feature_names) == 22
    assert "status" not in voice_dataset.feature_names
    assert "name" not in voice_dataset.feature_names


def test_voice_labels_harmonized(voice_dataset):
    assert set(voice_dataset.y.unique()) == {0, 1}
    # official class balance: 147 PD, 48 healthy
    assert voice_dataset.class_counts == {0: 48, 1: 147}


def test_voice_metadata(voice_dataset):
    md = voice_dataset.metadata
    assert md["recording_id"].is_unique
    assert md["voice_subject"].notna().all()
    # recordings grouped into a small set of subjects
    assert 25 <= md["voice_subject"].nunique() <= 35


def test_voice_features_numeric_and_finite(voice_dataset):
    X = voice_dataset.X
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)
    # spread1/spread2 are signed nonlinear measures; everything is finite
    assert not X.isna().any().any()


def test_voice_groups_parse_to_subjects(voice_dataset):
    first = voice_dataset.metadata["recording_id"].iloc[0]
    assert first.startswith("phon_R01_")
    assert voice_dataset.groups.iloc[0].startswith("S")
