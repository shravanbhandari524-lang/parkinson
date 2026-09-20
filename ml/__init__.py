"""Explainable Multimodal AI Framework for Early Parkinson's Disease Risk Prediction.

Machine-learning component: voice, handwriting and gait biomarkers.

Subpackages
-----------
data            dataset loaders and preprocessing
features        feature engineering (gait VGRF time series)
models          CV splits, per-modality training, fusion, metrics
inference       unified prediction API (probability, class, contributions)
explainability  SHAP for tree models, TabNet attention analysis
tests           pytest suite
artifacts       trained model artifacts (joblib / torch)
metrics         JSON metric reports
"""

__version__ = "0.1.0"
