# Explainable Multimodal AI Framework for Early Parkinson's Disease Risk Prediction

Machine-learning component: voice (UCI Parkinson's), handwriting (HandPD
spiral + meander) and gait (PhysioNet VGRF) biomarkers, with score-level
fusion and explainability.

All reported numbers below are **out-of-fold cross-validated results computed
on the real datasets** — nothing is synthetic or extrapolated.

## Results (mean over CV folds, threshold 0.5)

| Modality | Model | Role | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|---|
| Voice | XGBoost | primary | 0.887 | 0.918 | 0.793 | **0.959** |
| Voice | Random Forest | baseline | 0.913 | 0.938 | 0.836 | 0.957 |
| Voice | Logistic Regression | baseline | 0.805 | 0.816 | 0.773 | 0.906 |
| Handwriting | Random Forest | primary | 0.792 | 0.874 | 0.434 | **0.754** |
| Handwriting | SVM (RBF) | baseline | 0.816 | 0.989 | 0.119 | 0.730 |
| Handwriting | Logistic Regression | baseline | 0.681 | 0.660 | 0.749 | 0.760 |
| Gait | XGBoost | primary | 0.755 | 0.794 | 0.663 | 0.793 |
| Gait | Random Forest | baseline | 0.758 | 0.790 | 0.687 | **0.832** |
| Fusion | Weighted soft voting | baseline | 0.798 | 0.864 | 0.578 | 0.805 |
| Fusion | LR stacking | baseline | 0.759 | 0.785 | 0.673 | 0.804 |
| Fusion | **TabNet (proposed)** | proposed | 0.775 | 0.879 | 0.427 | 0.763 |

Notes on interpretation (important for honest reporting):

* **Voice** uses recording-level StratifiedKFold per the project spec. The UCI
  dataset contains multiple recordings per subject, so these figures are
  optimistic relative to a subject-level split (a subject-level protocol would
  be a stricter, recommended sensitivity analysis).
* **Handwriting** and **gait** are grouped by patient/subject (GroupKFold), so
  no patient/subject is ever in both train and validation folds. HandPD
  figures are therefore lower than the >90 % numbers often quoted in papers
  that split records randomly — those leak patients across folds.
* The fusion models operate on **out-of-fold** per-modality probabilities
  (see *Fusion design* below); they are not comparable 1:1 with
  single-modality numbers and reflect the difficulty of the pooled,
  modality-heterogeneous task.
* TabNet is the proposed multimodal fusion model. It was **not identified in
  the supplied literature matrix** — no claim of global novelty is made.

## Setup

Python **3.11** (a `.venv` is already created under `ml/.venv`).

```bash
py -3.11 -m venv ml/.venv
ml/.venv/Scripts/python.exe -m pip install -r ml/requirements.txt
```

Configuration lives in `ml/config.yaml` (paths relative to the repo root and
`ml/`). Override without editing the file:

* copy `ml/config.yaml` to `ml/config.local.yaml` (git-ignored), or
* set environment variables: `PARK_ML_DATA_DIR`, `PARK_ML_VOICE_PATH`,
  `PARK_ML_HANDWRITING_SPIRAL_PATH`, `PARK_ML_HANDWRITING_MEANDER_PATH`,
  `PARK_ML_GAIT_PATH`, `PARK_ML_ARTIFACTS_DIR`, `PARK_ML_METRICS_DIR`,
  `PARK_ML_SEED`.

Datasets are read from `Parkinson/` (git-ignored data directory):

* `Parkinson/Voice/parkinsons.data` — UCI Parkinson's, 195 recordings × 22
  features, `status` 1 = PD / 0 = healthy.
* `Parkinson/Handwriting/archive (7)/{Spiral,Meander}_HandPD.csv` — HandPD,
  368 rows each, `CLASS_TYPE` 2 = PD / 1 = healthy (converted to 1/0).
  Grouping key is `(CLASS_TYPE, ID_PATIENT)` — the raw `ID_PATIENT` numbering
  restarts per class (healthy 1–18, PD 1–37 with id 4 absent), so it is
  ambiguous on its own.
* `Parkinson/Gait/gait-in-parkinsons-disease-1.0.0/...` — PhysioNet gait,
  306 records (214 PD / 92 control) from 165 subjects across the disjoint
  Ga/Si/Ju cohorts; `GaPt03` and `JuPt03` are different people, so the full
  prefixed id is the subject key.

## Usage

```bash
# full pipeline: per-modality training -> fusion -> TabNet -> SHAP summaries
ml/.venv/Scripts/python.exe ml/train_all.py

# skip stages (repeatable): voice handwriting gait fusion explain
ml/.venv/Scripts/python.exe ml/train_all.py --skip explain

# tests (unit tests are hermetic; two integration tests skip until
# ml/train_all.py has produced artifacts)
ml/.venv/Scripts/python.exe -m pytest
```

Outputs:

* `ml/artifacts/` — joblib pipelines per model, `fusion_tabnet.zip`,
  fusion combiners.
* `ml/metrics/` — `voice_metrics.json`, `handwriting_metrics.json`,
  `gait_metrics.json`, `fusion_metrics.json`, `tabnet_fusion_metrics.json`
  (full metric suite: accuracy, precision, recall, F1, sensitivity,
  specificity, ROC-AUC, confusion matrix; per-fold and aggregated; plus
  library versions and UTC timestamps), `oof_*.csv` out-of-fold
  probabilities, `fusion_meta_dataset.csv`, `shap_summary_*.json`.

## Moving to another laptop (GitHub workflow)

What git carries and what it deliberately does not:

| | Travels via git | Must be recreated on the new machine |
|---|---|---|
| Source code, tests, `requirements.txt`, `config.yaml`, `.gitignore` | ✅ | — |
| `Parkinson/` datasets (2.4 GB, licence/size) | ❌ ignored | copy the folder (USB/Drive) or download from the sources below |
| `ml/.venv/` (1.2 GB, OS-specific) | ❌ ignored | `py -3.11 -m venv ml/.venv` + `pip install -r ml/requirements.txt` |
| `ml/artifacts/` models (11 MB) | ❌ ignored | regenerated by `ml/train_all.py` in ~1 min — or force-add them (below) |
| `ml/metrics/` reports (JSON/CSV) | ❌ ignored | regenerated together with the artifacts |

Step-by-step on the new machine:

```bash
# 1. install Python 3.11 (https://www.python.org/downloads/), then verify
py -3.11 --version

# 2. clone and set up the environment
git clone https://github.com/<you>/<repo>.git
cd <repo>
py -3.11 -m venv ml/.venv
ml/.venv/Scripts/python.exe -m pip install -r ml/requirements.txt

# 3. place the datasets (exact layout expected by ml/config.yaml):
#    Parkinson/Voice/parkinsons.data
#    Parkinson/Handwriting/archive (7)/Spiral_HandPD.csv
#    Parkinson/Handwriting/archive (7)/Meander_HandPD.csv
#    Parkinson/Gait/gait-in-parkinsons-disease-1.0.0/gait-in-parkinsons-disease-1.0.0/  (record files + demographics)
#    Data sources: UCI Parkinson's dataset; HandPD (Kaggle);
#    PhysioNet "Gait in Parkinson's Disease" 1.0.0 (physionet.org)
#    If the data lives somewhere else, set PARK_ML_DATA_DIR instead of moving it.

# 4. verify: all 67 tests pass (loader/feature tests need the data present)
ml/.venv/Scripts/python.exe -m pytest

# 5. train everything and regenerate artifacts + metrics (~1 min)
ml/.venv/Scripts/python.exe ml/train_all.py
```

Optional — ship the trained models through git so the new laptop can run
inference without retraining (they are git-ignored by default):

```bash
git add -f ml/artifacts/*.joblib ml/artifacts/fusion_tabnet.zip
```

Linux/macOS note: use `ml/.venv/bin/python` instead of `ml/.venv/Scripts/python.exe`;
everything else is identical (pure-Python deps; torch/xgboost ship wheels for
all three OSes).

## Architecture

```
ml/
├── config.py, config.yaml     # env-var + file config, no hard-coded paths
├── data/
│   ├── loaders.py             # voice / handwriting / gait loaders (harmonised labels)
│   └── preprocessing.py       # per-fold imputation/scaling inside sklearn Pipelines
├── features/
│   └── gait.py                # reusable VGRF time/frequency feature extraction
├── models/
│   ├── common.py              # CV training engine, model factories, persistence
│   ├── splits.py              # StratifiedKFold / GroupKFold / LOGO + leakage guards
│   ├── metrics.py -> ml/metrics/metrics.py  # metric suite + JSON reports
│   ├── voice.py / handwriting.py / gait.py  # modality entry points
│   ├── fusion.py              # score-level fusion: weighted soft voting, LR stacking
│   └── tabnet_model.py        # proposed PyTorch TabNet fusion
├── explainability/
│   ├── shap_explainer.py      # SHAP TreeExplainer (instance + global)
│   └── tabnet_attention.py    # TabNet attention masks -> feature/modality contributions
├── inference/
│   └── predict.py             # MultimodalPredictor (explained predictions)
├── metrics/                   # generated metric JSON / OOF tables
├── artifacts/                 # generated model artifacts
└── tests/                     # 67 tests (unit + gated integration)
```

### Data handling and leakage policy

* Labels are harmonised everywhere to **1 = Parkinson's, 0 = healthy**.
* Split strategies: voice → `StratifiedKFold` (recording-level, per spec);
  handwriting → `GroupKFold` on `(CLASS_TYPE, ID_PATIENT)`; gait →
  `GroupKFold` on subject id. `iter_folds` asserts per fold that no
  patient/subject group appears on both sides (the assertion is intentionally
  disabled only for the voice recording-level design, which is documented in
  the metrics reports).
* Preprocessing (imputation, scaling) lives inside the sklearn pipelines and
  is refit per training fold — no statistics from validation data leak into
  training.
* The fusion stage consumes only **out-of-fold** per-modality probabilities,
  and its own CV is grouped by patient/subject as well.

### Gait feature extraction (`ml/features/gait.py`)

Per record (100 Hz, 8 left + 8 right VGRF sensors, reusable pure-numpy
functions): contact detection per foot → cadence, stride-time mean/std/CV,
stance time, swing time, stance/swing ratio, left–right symmetry; force
mean/max/std/CV and left–right force asymmetry; frequency domain: dominant
frequency, spectral entropy, low-frequency power ratio; plus statistical
features (mean/std/CV/skewness/kurtosis/RMS of the total-force channels).
Timing features exclude boundary-truncated contacts (records commonly start
or end mid-stride, which would otherwise bias stride/stance statistics).

### Fusion design

The three source datasets are **disjoint cohorts** — no patient exists in
more than one modality — so fusion operates at *score level*: each meta-row
carries its own modality's out-of-fold primary-model probability, neutral
fills (0.5) and missing indicators for the other two. The combiners
(weighted soft voting with weights fitted by a leakage-safe simplex grid
search, LR stacking, and the proposed TabNet) therefore learn a rule that
also works when a new patient is measured on a subset of modalities. The
meta-dataset is written to `ml/metrics/fusion_meta_dataset.csv` for audit.

### Explainability

* **Tree models** — exact SHAP `TreeExplainer` values: signed per-feature
  contributions per prediction and mean-|SHAP| global summaries
  (`ml/metrics/shap_summary_*.json`).
* **TabNet** — sparse attention masks aggregated to per-feature importances
  and grouped into **modality contributions** (voice / handwriting / gait,
  normalised to sum to 1).
* `MultimodalPredictor.predict(voice=..., handwriting=..., gait=...)` returns
  probability, predicted class, per-modality probabilities, top features,
  feature contributions and modality contributions. Gait accepts a raw
  `(n, 19)` VGRF array and featurizes it automatically.

```python
from ml.inference import MultimodalPredictor

predictor = MultimodalPredictor()
result = predictor.predict(voice={"MDVP:Fo(Hz)": 152.1, ...}, gait=np.loadtxt("GaPt03_Ga01_01.txt"))
```

### Novelty statement

TabNet is used as the proposed multimodal fusion model of this framework. It
was not identified in the supplied literature matrix; no claim of global
novelty is made.
