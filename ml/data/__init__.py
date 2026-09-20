"""Dataset loaders and preprocessing for the three modalities."""

from ml.data.loaders import (
    Dataset,
    load_gait,
    load_gait_record,
    list_gait_records,
    load_handwriting,
    load_voice,
)
from ml.data.preprocessing import (
    drop_correlated_features,
    make_preprocessor,
    select_numeric_features,
)

__all__ = [
    "Dataset",
    "load_voice",
    "load_handwriting",
    "load_gait",
    "load_gait_record",
    "list_gait_records",
    "drop_correlated_features",
    "make_preprocessor",
    "select_numeric_features",
]
