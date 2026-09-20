"""Explainability: SHAP for tree models, attention analysis for TabNet."""

from ml.explainability.shap_explainer import (
    global_importance,
    instance_contributions,
    make_tree_explainer,
    summarize_modality,
)
from ml.explainability.tabnet_attention import (
    attention_with_names,
    global_feature_importance,
    load_tabnet,
    modality_contributions,
)

__all__ = [
    "make_tree_explainer",
    "instance_contributions",
    "global_importance",
    "summarize_modality",
    "load_tabnet",
    "attention_with_names",
    "modality_contributions",
    "global_feature_importance",
]
