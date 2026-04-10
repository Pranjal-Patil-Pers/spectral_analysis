# Spectral Analysis — SEP Event Prediction

## Domain & Scientific Goal

**Solar Energetic Particle (SEP) Event Prediction** — a space weather forecasting problem. The project trains machine learning models on multi-channel particle flux time series to predict solar energetic particle events at varying lead times (5–120 minutes ahead).

---

## Project Structure

```
spectral_analysis/
├── notebooks/          # All experiment scripts + data_analysis.py
├── data/
│   ├── raw/            # 2,895 CSV files (1-min cadence time series)
│   ├── SEP_class_labels.csv   # File → Label mapping
│   └── reports/        # All outputs, one subdirectory per experiment
├── scripts/            # experiment1.py (early prototype)
├── requirements.txt
└── Dockerfile
```

---

## Data

- **2,895 CSV files**, each a time series with 4 particle flux channels:
  - `p3_flux_ic` (proton 3–6 MeV), `p5_flux_ic` (5–10 MeV), `p7_flux_ic` (7–10 MeV), `long` (X-ray)
- **Labels**: binary (SEP event or not), stored in `SEP_class_labels.csv`
- **Temporal split** (year-based to avoid leakage):
  - Train: ≤ 1992 | Val: 1992–2002 | Test: 2002–2018
- **Event anchoring**: Each series is aligned to an `event_onset_index = 720` (minute 720 = 12 h into the series)
- **Observation window**: 360-minute (6h) lookback before the event onset

---

## Core Pipeline

### 1. Data Loading — `TimeSeriesDataset`
Reads CSVs via label file, parses event timestamps from filenames (`YYYY-MM-DD_HH-MM.csv`), applies `LabelEncoder`.

### 2. Feature Engineering — FFT
- Extract a 360-min window shifted back by the forecast lag: `[event - lag - 360 : event - lag]`
- Divide into non-overlapping slices by FFT window size (45, 90, 180, 360 min)
- Apply `scipy.rfft` per slice per channel → extract **magnitude + phase**
- Final feature matrix shape: `(N_samples, 2×channels × n_slices × max_coefficients)`

| Window size | Slices (360/win) | Max coeffs | Total raw features |
|---|---|---|---|
| 45 min | 8 | 23 | ~3,680 |
| 90 min | 4 | 46 | ~3,680 |
| 180 min | 2 | 91 | ~3,640 |
| 360 min | 1 | 181 | ~1,448 |

### 3. Model — RandomForestClassifier
- 300 trees, `random_state=42`
- No hyperparameter tuning; focus is on feature selection and transformation

### 4. Evaluation Metrics
- Standard: Accuracy, F1, Precision, Recall
- Solar-specific:
  - **TSS** (True Skill Statistic) = Sensitivity + Specificity − 1
  - **HSS** (Heidke Skill Score) = skill relative to random chance
  - **CSS** (Critical Success Index) = TP / (TP + FP + FN)
- **Model selection priority**: val_css → val_tss → val_hss → val_f1 → fewer features

---

## Experiment Evolution

### Experiment 12 — FFT Coefficient Sweep
**Question:** How many FFT coefficients are actually needed?
- Sweeps k = 1 … max_coeffs for every (lag, window) pair
- Finds the "elbow" — smallest k within tolerance of best validation CSS
- **Output**: Sweep CSVs, elbow curves, calibration plots

### Experiment 13 — Lag-Level Concatenation
**Question:** Can we combine window features per lag using Exp12's selections?
- Loads Exp12 result logs, selects best k per window
- Concatenates all windows' selected features into one lag-level feature matrix
- Trains one RF per lag
- **Output**: Per-lag results, feature importance grids

### Experiment 14 — Per-Window Top-% Selection
**Question:** What if we use percentage-based instead of count-based selection?
- For each lag, trains baseline RF per window, ranks features by importance
- Selects top p% (5, 10, 15, 20, 25%) per window, then concatenates
- **Output**: Top-% sweep results, 5×5 importance grid plots

### Experiment 15 — Global Top-% Selection
**Question:** What if we rank features globally across all windows instead of per-window?
- Concatenates all windows first, trains global baseline RF per lag
- Selects top p% (5, 10, 15, 25, 50, 100%) from the global importance ranking
- Extensive feature importance analysis: channel timelines, cumulative importance, global explainability panels
- **Output**: Richer set of CSVs and plots; per-channel importance breakdowns

### Experiment 16 — Independent Window Top-% + Counterfactuals
**Question:** Can we explain model decisions via counterfactual time series?
- Trains independent RF per (lag, window, top-%) — no concatenation
- Selects best model overall and per lag
- Generates **DiCE counterfactual explanations** for 2 test samples per class
- Reconstructs original & counterfactual time series via **inverse FFT**
- Plots side-by-side original vs. counterfactual flux traces
- **Output**: Hundreds of pickled models, counterfactual time series plots, reconstruction error CSVs

### Experiment 17 — A/B/C Input Transformation Comparison
**Question:** Does exponentiating the raw flux before FFT improve performance?
- Runs the full Exp16 pipeline on three input variants:

| Variant | Preprocessing |
|---|---|
| `baseline` | Raw flux → FFT |
| `exp_fft` | Scale per-channel (max → 10) → exp → FFT |
| `exp_fft_no_scaling` | Clip at log(float32_max) ≈ 88.72 → exp → FFT |

- Produces identical artifacts per variant under `data/reports/experiment17/<variant>/`
- Final `experiment17_ab_compare_*.csv` summarises best model from each variant side-by-side

---

## Outputs per Experiment

| Artifact | Description |
|---|---|
| `*_results_*.csv` | Full grid-search results (lag × window × top-% × metrics) |
| `*_selection_*.csv` | Feature indices/metadata selected per config |
| `*_best_model_summary_*.csv` | Single best-model row |
| `*_val/test_css_by_lag/window_*.png` | Performance plots |
| `*_features_required_cumimp_gt50_*.png` | Features needed for 50% cumulative importance |
| `saved_models/*.pkl` | Pickled RF model + feature metadata per config |
| `counterfactual_timeseries/**/*.png` | Original vs. counterfactual flux plots (Exp16/17) |
| `*_counterfactual_summary_*.csv` | Reconstruction errors + distances per sample |

---

## Dependencies

| Library | Role |
|---|---|
| `scikit-learn` | RandomForest, metrics, LabelEncoder |
| `scipy.fft` | `rfft` / `irfft` for feature extraction & reconstruction |
| `dice-ml` | Counterfactual explanation generation |
| `pandas` / `numpy` | Data handling |
| `matplotlib` / `seaborn` | All plotting |
| `shap` | Feature attribution |

Install dependencies:

```bash
pip install -r requirements.txt
```
