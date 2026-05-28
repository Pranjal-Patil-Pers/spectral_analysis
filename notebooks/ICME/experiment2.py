"""
ICME Experiment 2 — FFT (magnitude + phase) features, Box-Cox transform,
Random Forest classification, DiCE counterfactual generation.

Data   : per-event CSVs from experiment1.py  (data/ICMECAT/)
Features: Box-Cox → rfft → [magnitude | phase]  (both components)
Model  : RandomForestClassifier  (top-50% cumulative importance features)
XAI    : DiCE random-sampling counterfactuals
Output : data/reports/experiment_icme2/
"""

from __future__ import annotations

import math
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import dice_ml

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "ICMECAT"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "ICMECAT" / "reports"

SERIES_LEN        = 48     # resample every event to this many hourly steps
N_TREES           = 300
RANDOM_STATE      = 42
NUM_CF            = 3      # counterfactuals requested per query
NUM_EXAMPLES_PER_CLASS = 5  # DiCE query examples per class

# ── Data loading ───────────────────────────────────────────────────────────────

def load_icme_dataset(
    data_dir: Path,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Return (X_raw, y, sample_ids) from labels.csv + per-event CSVs."""
    labels_df = pd.read_csv(data_dir / "labels.csv")
    X: list[np.ndarray] = []
    y: list[int] = []
    ids: list[str] = []

    for _, row in labels_df.iterrows():
        csv_path = data_dir / str(row["file"])
        if not csv_path.exists():
            continue
        series = pd.read_csv(csv_path)["icme_speed_mean"].to_numpy(dtype=np.float64)
        series = series[np.isfinite(series)]
        if len(series) < 4:
            continue
        X.append(series)
        y.append(int(row["label"]))
        ids.append(csv_path.stem)

    return X, np.array(y, dtype=np.int32), ids


def resample_series(series: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-interpolation resample to target_len."""
    n = len(series)
    if n == target_len:
        return series.copy()
    return np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, n),
        series,
    )


# ── Box-Cox transform ──────────────────────────────────────────────────────────

def fit_box_cox(X_train: list[np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate(X_train).ravel()
    flat = flat[np.isfinite(flat)]
    min_val = float(np.min(flat))
    shift = 1e-6 if min_val > 0 else (-min_val + 1e-6)
    _, lambda_ = stats.boxcox(flat + shift)
    return float(shift), float(lambda_)


def apply_box_cox(series: np.ndarray, shift: float, lam: float) -> np.ndarray:
    return stats.boxcox(series + shift, lmbda=lam).astype(np.float64)


def invert_box_cox(series: np.ndarray, shift: float, lam: float) -> np.ndarray:
    return (inv_boxcox(series.astype(np.float64), lam) - shift).astype(np.float64)


# ── FFT features (magnitude + phase) ──────────────────────────────────────────

def extract_fft_features(series: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Real FFT of series → concatenate [magnitude, phase].

    magnitude[k] = |X[k]|          (always ≥ 0)
    phase[k]     = angle(X[k])     (radians, [-π, π])
    """
    coeffs    = np.fft.rfft(series)
    magnitude = np.abs(coeffs)
    phase     = np.angle(coeffs)
    features  = np.concatenate([magnitude, phase])
    n         = len(coeffs)
    names     = [f"mag_coeff_{k}" for k in range(n)] + [f"phase_coeff_{k}" for k in range(n)]
    return features.astype(np.float64), names


def build_feature_matrix(
    X_raw: list[np.ndarray],
    series_len: int,
    shift: float,
    lam: float,
) -> tuple[np.ndarray, list[str]]:
    """Resample → Box-Cox → FFT for every series. Returns (N, F) and feature names."""
    rows: list[np.ndarray] = []
    names: list[str] = []
    for s in X_raw:
        resampled   = resample_series(s, series_len)
        transformed = apply_box_cox(resampled, shift, lam)
        feats, feat_names = extract_fft_features(transformed)
        rows.append(feats)
        if not names:
            names = feat_names
    return np.array(rows, dtype=np.float64), names


# ── Reconstruction (CF features → time series) ────────────────────────────────

def reconstruct_from_fft_features(
    features: np.ndarray,
    series_len: int,
    shift: float,
    lam: float,
) -> np.ndarray:
    """Magnitude + phase features → IFFT → invert Box-Cox → km/s."""
    n_coeffs  = series_len // 2 + 1
    magnitude = features[:n_coeffs]
    phase     = features[n_coeffs : 2 * n_coeffs]
    coeffs    = magnitude * np.exp(1j * phase)
    bc_signal = np.fft.irfft(coeffs, n=series_len)
    return invert_box_cox(bc_signal, shift, lam)


# ── DiCE helpers ───────────────────────────────────────────────────────────────

class _SelectedFeatureRF:
    """sklearn-compatible wrapper: accepts selected-feature inputs, routes to full RF."""

    def __init__(self, rf: RandomForestClassifier, selected_idx: np.ndarray):
        self._rf   = rf
        self._idx  = selected_idx
        self.classes_ = rf.classes_

    def _expand(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float64)
        full  = np.zeros((X_arr.shape[0], self._rf.n_features_in_), dtype=np.float64)
        full[:, self._idx] = X_arr
        return full

    def predict(self, X) -> np.ndarray:
        return self._rf.predict(self._expand(X))

    def predict_proba(self, X) -> np.ndarray:
        return self._rf.predict_proba(self._expand(X))


def fit_dice_metric_stats(X_train_sel: np.ndarray) -> dict[str, np.ndarray]:
    median  = np.median(X_train_sel, axis=0)
    mad     = np.median(np.abs(X_train_sel - median), axis=0)
    mad     = np.where(mad > 0, mad, 1.0)
    weights = np.where(np.isfinite(1.0 / mad), 1.0 / mad, 1.0)
    return {"feature_mad": mad, "feature_weights": weights}


def dice_proximity(
    original: np.ndarray,
    cf: np.ndarray,
    metric_stats: dict[str, np.ndarray],
) -> float:
    return float(np.mean(np.abs(original - cf) * metric_stats["feature_weights"]))


def _weighted_l1(
    x: np.ndarray,
    y: np.ndarray,
    metric_stats: dict[str, np.ndarray],
    average: bool = True,
) -> float:
    w = metric_stats["feature_weights"]
    d = float(np.sum(np.abs(x - y) * w))
    return d / len(w) if (average and len(w) > 0) else d


def compute_cf_metrics(
    original_sel: np.ndarray,
    cf_sel: np.ndarray,
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    cf_predicted_label: int,
    metric_stats: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Proximity, sparsity, and plausibility for one counterfactual."""
    # ── Proximity ────────────────────────────────────────────────────────────
    proximity_loss = _weighted_l1(original_sel, cf_sel, metric_stats, average=True)

    # ── Sparsity ─────────────────────────────────────────────────────────────
    changed_mask  = np.abs(cf_sel - original_sel) > 1e-9
    n_changed     = int(np.sum(changed_mask))
    n_total       = len(original_sel)
    sparsity_loss = n_changed / n_total if n_total > 0 else np.nan

    # ── Plausibility (distance to nearest same-class training example) ────────
    same_class = y_train == cf_predicted_label
    pool       = X_train_sel[same_class] if np.any(same_class) else X_train_sel
    distances  = np.array([
        _weighted_l1(cf_sel, pool[j], metric_stats, average=True)
        for j in range(len(pool))
    ])
    nearest_idx        = int(np.argmin(distances))
    plausibility_dist  = float(distances[nearest_idx])

    return {
        "changed_feature_count":     n_changed,
        "changed_feature_fraction":  sparsity_loss,
        "dice_proximity_loss":       proximity_loss,
        "dice_proximity_score":      1.0 / (1.0 + proximity_loss),
        "dice_sparsity_loss":        sparsity_loss,
        "dice_sparsity_score":       float(1.0 - sparsity_loss) if np.isfinite(sparsity_loss) else np.nan,
        "dice_plausibility_distance": plausibility_dist,
        "dice_plausibility_score":   1.0 / (1.0 + plausibility_dist),
    }


def compute_diversity_metrics(
    cf_vectors: list[np.ndarray],
    metric_stats: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """DPP determinant + average pairwise distance over the full CF set."""
    if len(cf_vectors) < 2:
        return {
            "dice_diversity_dpp":           0.0,
            "dice_diversity_avg_dist":      0.0,
            "dice_diversity_pair_count":    0,
            "dice_mean_pairwise_distance":  0.0,
        }

    cfs = np.array(cf_vectors, dtype=np.float64)
    n   = len(cfs)
    kernel = np.zeros((n, n), dtype=np.float64)
    pairwise: list[float] = []

    for i in range(n):
        for j in range(n):
            d   = _weighted_l1(cfs[i], cfs[j], metric_stats, average=False)
            sim = 1.0 / (1.0 + d)
            kernel[i, j] = sim + (0.0001 if i == j else 0.0)
            if i < j:
                pairwise.append(d)

    avg_sim = float(np.mean([1.0 / (1.0 + d) for d in pairwise])) if pairwise else 1.0
    return {
        "dice_diversity_dpp":          float(np.linalg.det(kernel)),
        "dice_diversity_avg_dist":     float(1.0 - avg_sim),
        "dice_diversity_pair_count":   len(pairwise),
        "dice_mean_pairwise_distance": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def summarize_metrics(cf_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-CF metrics into a one-row summary table."""
    found = cf_summary[cf_summary["counterfactual_found"].astype(bool)]
    n_found    = len(found)
    n_total    = len(cf_summary)
    summary: dict[str, float | int] = {
        "total_queries":          n_total,
        "counterfactuals_found":  n_found,
        "success_rate":           n_found / n_total if n_total > 0 else np.nan,
    }
    scalar_metrics = [
        "dice_proximity_loss",   "dice_proximity_score",
        "dice_sparsity_loss",    "dice_sparsity_score",
        "dice_plausibility_distance", "dice_plausibility_score",
        "changed_feature_count", "changed_feature_fraction",
        "reconstruction_mse",
    ]
    for col in scalar_metrics:
        vals = pd.to_numeric(found[col], errors="coerce").dropna() if col in found else pd.Series(dtype=float)
        summary[f"mean_{col}"]   = float(vals.mean())   if len(vals) > 0 else np.nan
        summary[f"median_{col}"] = float(vals.median()) if len(vals) > 0 else np.nan

    diversity_cols = [
        "dice_diversity_dpp", "dice_diversity_avg_dist",
        "dice_diversity_pair_count", "dice_mean_pairwise_distance",
    ]
    for col in diversity_cols:
        vals = pd.to_numeric(found[col], errors="coerce").dropna() if col in found else pd.Series(dtype=float)
        summary[col] = float(vals.iloc[0]) if len(vals) > 0 else np.nan

    return pd.DataFrame([summary])


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_cf_timeseries(
    sample_id: str,
    original: np.ndarray,
    cf_recon: np.ndarray,
    label_name: str,
    target_label_name: str,
    mse: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(original))
    ax.plot(x, original, color="#1f77b4", linewidth=1.8, label="original")
    ax.plot(x, cf_recon,  color="#ff7f0e", linewidth=1.8, linestyle="--", label="counterfactual")
    ax.set_xlabel("Time step (hours)")
    ax.set_ylabel("Solar wind speed (km/s)")
    ax.set_title(
        f"{sample_id}  |  {label_name} → {target_label_name}  |  MSE={mse:.3e}",
        fontsize=11,
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cf_gallery(
    plot_records: list[dict],
    output_path: Path,
    title: str = "ICME Counterfactual Gallery",
) -> None:
    if not plot_records:
        return
    n     = min(len(plot_records), 6)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for i, rec in enumerate(plot_records[:n]):
        ax = axes_flat[i]
        x  = np.arange(len(rec["original"]))
        ax.plot(x, rec["original"], color="#1f77b4", lw=1.6, label="original")
        ax.plot(x, rec["cf_recon"],  color="#ff7f0e", lw=1.6, ls="--", label="counterfactual")
        ax.set_title(
            f"{rec['sample_id']}  {rec['label_name']} → {rec['target_label_name']}\n"
            f"MSE={rec['mse']:.3e}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cf_ts_dir = OUTPUT_DIR / "counterfactual_timeseries" / run_stamp
    cf_ts_dir.mkdir(parents=True, exist_ok=True)

    classes = np.array(["quiet", "icme"])

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading ICME dataset …")
    X_raw, y, ids = load_icme_dataset(DATA_DIR)
    print(f"  {len(X_raw)} events — ICME: {(y==1).sum()}, quiet: {(y==0).sum()}")

    # ── 2. Stratified 70/15/15 split ─────────────────────────────────────────
    idx = np.arange(len(X_raw))
    idx_tv, idx_test, y_tv, y_test = train_test_split(
        idx, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=RANDOM_STATE
    )
    X_train_raw = [X_raw[i] for i in idx_train]
    X_val_raw   = [X_raw[i] for i in idx_val]
    X_test_raw  = [X_raw[i] for i in idx_test]
    ids_train   = [ids[i] for i in idx_train]
    ids_test    = [ids[i] for i in idx_test]
    print(f"  Train {len(X_train_raw)} / Val {len(X_val_raw)} / Test {len(X_test_raw)}")

    # ── 3. Fit Box-Cox on training set ────────────────────────────────────────
    print("Fitting Box-Cox transform …")
    shift, lam = fit_box_cox(X_train_raw)
    print(f"  shift={shift:.6f}  lambda={lam:.4f}")

    # ── 4. Build FFT feature matrices (magnitude + phase) ─────────────────────
    print(f"Extracting FFT features (magnitude + phase, series_len={SERIES_LEN}) …")
    X_train_feat, feature_names = build_feature_matrix(X_train_raw, SERIES_LEN, shift, lam)
    X_val_feat,   _             = build_feature_matrix(X_val_raw,   SERIES_LEN, shift, lam)
    X_test_feat,  _             = build_feature_matrix(X_test_raw,  SERIES_LEN, shift, lam)
    print(f"  Shape: {X_train_feat.shape}  ({len(feature_names)} features)")

    # ── 5. Train Random Forest on all features ────────────────────────────────
    print("Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_feat, y_train)
    val_acc  = rf.score(X_val_feat,  y_val)
    test_acc = rf.score(X_test_feat, y_test)
    print(f"  Val accuracy : {val_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(classification_report(y_test, rf.predict(X_test_feat), target_names=["quiet", "icme"]))

    # ── 6. Select top-50% cumulative-importance features ─────────────────────
    importances = rf.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    cumsum      = np.cumsum(importances[sorted_idx])
    n_top       = int(np.searchsorted(cumsum, 0.50)) + 1
    selected_idx   = sorted_idx[:n_top]
    selected_names = [feature_names[i] for i in selected_idx]
    n_mag_sel   = sum(1 for n in selected_names if n.startswith("mag_"))
    n_phase_sel = sum(1 for n in selected_names if n.startswith("phase_"))
    print(f"  Top-50% features: {n_top}/{len(feature_names)}  "
          f"(magnitude: {n_mag_sel}, phase: {n_phase_sel})")

    X_train_sel = X_train_feat[:, selected_idx]
    X_test_sel  = X_test_feat[:,  selected_idx]

    # ── 7. Save model artifact ────────────────────────────────────────────────
    model_path = OUTPUT_DIR / f"icme2_rf_{run_stamp}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({
            "model":                    rf,
            "feature_names":            feature_names,
            "selected_feature_names":   selected_names,
            "selected_feature_indices": selected_idx.tolist(),
            "shift":                    shift,
            "lambda_":                  lam,
            "series_len":               SERIES_LEN,
            "val_accuracy":             val_acc,
            "test_accuracy":            test_acc,
        }, fh)
    print(f"  Model saved → {model_path}")

    # ── 8. DiCE setup ─────────────────────────────────────────────────────────
    print("Setting up DiCE …")
    metric_stats = fit_dice_metric_stats(X_train_sel)

    train_df = pd.DataFrame(X_train_sel, columns=selected_names)
    train_df["label"] = y_train.astype(int)

    wrapped_model = _SelectedFeatureRF(rf, selected_idx)

    dice_data  = dice_ml.Data(
        dataframe=train_df,
        continuous_features=selected_names,
        outcome_name="label",
    )
    dice_model = dice_ml.Model(model=wrapped_model, backend="sklearn")
    exp_dice   = dice_ml.Dice(dice_data, dice_model, method="random")

    # ── 9. Generate counterfactuals ───────────────────────────────────────────
    print("Generating counterfactuals …")
    cf_rows: list[dict]      = []
    plot_records: list[dict] = []

    for orig_label in [0, 1]:
        target_label      = 1 - orig_label
        label_name        = str(classes[orig_label])
        target_label_name = str(classes[target_label])

        mask     = y_test == orig_label
        cand_idx = np.where(mask)[0]
        preds    = rf.predict(X_test_feat[cand_idx])
        correct  = cand_idx[preds == orig_label]
        pool     = correct if len(correct) >= NUM_EXAMPLES_PER_CLASS else cand_idx
        chosen   = pool[:NUM_EXAMPLES_PER_CLASS]

        for local_i in chosen:
            sample_id     = ids_test[local_i]
            query_df      = pd.DataFrame([X_test_sel[local_i]], columns=selected_names)
            cf_generated  = False
            mse           = np.nan
            error         = ""
            best_metrics: dict[str, float | int] = {}
            diversity_metrics_inst: dict[str, float | int] = compute_diversity_metrics([], metric_stats)

            try:
                cf_result = exp_dice.generate_counterfactuals(
                    query_df,
                    total_CFs=NUM_CF,
                    desired_class=int(target_label),
                    features_to_vary=selected_names,
                    verbose=False,
                    sample_size=2000,
                    random_seed=RANDOM_STATE + int(local_i),
                )
                cf_df = cf_result.cf_examples_list[0].final_cfs_df
                if len(cf_df) == 0:
                    raise RuntimeError("DiCE returned no counterfactual rows.")

                # ── evaluate every returned CF ────────────────────────────
                instance_cf_vectors: list[np.ndarray] = []
                all_per_cf_metrics: list[dict] = []

                for _, cf_row in cf_df.iterrows():
                    cf_vals = cf_row[selected_names].astype(float).to_numpy(dtype=np.float64)
                    cf_full_tmp = X_test_feat[local_i].copy()
                    cf_full_tmp[selected_idx] = cf_vals
                    cf_pred = int(rf.predict(cf_full_tmp.reshape(1, -1))[0])
                    m = compute_cf_metrics(
                        original_sel=X_test_sel[local_i],
                        cf_sel=cf_vals,
                        X_train_sel=X_train_sel,
                        y_train=y_train,
                        cf_predicted_label=cf_pred,
                        metric_stats=metric_stats,
                    )
                    instance_cf_vectors.append(cf_vals)
                    all_per_cf_metrics.append(m)

                # ── best CF = lowest proximity loss (closest to original) ──
                best_idx     = int(np.argmin([m["dice_proximity_loss"] for m in all_per_cf_metrics]))
                best_cf_vals = instance_cf_vectors[best_idx]
                best_metrics = all_per_cf_metrics[best_idx]

                # ── per-instance diversity over all NUM_CF CFs ────────────
                diversity_metrics_inst = compute_diversity_metrics(instance_cf_vectors, metric_stats)

                # ── reconstruct best CF → km/s for plotting ───────────────
                cf_full = X_test_feat[local_i].copy()
                cf_full[selected_idx] = best_cf_vals
                original_resampled = resample_series(X_test_raw[local_i], SERIES_LEN)
                cf_recon = reconstruct_from_fft_features(cf_full, SERIES_LEN, shift, lam)
                mse = float(np.mean((original_resampled - cf_recon) ** 2))

                cf_generated = True
                plot_records.append({
                    "sample_id":         sample_id,
                    "original":          original_resampled,
                    "cf_recon":          cf_recon,
                    "label_name":        label_name,
                    "target_label_name": target_label_name,
                    "mse":               mse,
                })
                plot_cf_timeseries(
                    sample_id, original_resampled, cf_recon,
                    label_name, target_label_name, mse,
                    cf_ts_dir / f"cf_{sample_id}_{label_name}_to_{target_label_name}.png",
                )
            except Exception as exc:
                error = str(exc)

            cf_rows.append({
                "sample_id":            sample_id,
                "original_label":       orig_label,
                "original_label_name":  label_name,
                "target_label":         target_label,
                "target_label_name":    target_label_name,
                "counterfactual_found": cf_generated,
                "reconstruction_mse":   mse,
                # proximity + sparsity + plausibility on best CF
                "changed_feature_count":      best_metrics.get("changed_feature_count",      np.nan),
                "changed_feature_fraction":   best_metrics.get("changed_feature_fraction",   np.nan),
                "dice_proximity_loss":        best_metrics.get("dice_proximity_loss",        np.nan),
                "dice_proximity_score":       best_metrics.get("dice_proximity_score",       np.nan),
                "dice_sparsity_loss":         best_metrics.get("dice_sparsity_loss",         np.nan),
                "dice_sparsity_score":        best_metrics.get("dice_sparsity_score",        np.nan),
                "dice_plausibility_distance": best_metrics.get("dice_plausibility_distance", np.nan),
                "dice_plausibility_score":    best_metrics.get("dice_plausibility_score",    np.nan),
                # per-instance diversity over all NUM_CF CFs
                **diversity_metrics_inst,
                "error": error,
            })

    # ── 10. Save results ──────────────────────────────────────────────────────
    cf_summary   = pd.DataFrame(cf_rows)
    summary_path = OUTPUT_DIR / f"icme2_counterfactual_summary_{run_stamp}.csv"
    cf_summary.to_csv(summary_path, index=False)

    metrics_df   = summarize_metrics(cf_summary)
    metrics_path = OUTPUT_DIR / f"icme2_counterfactual_metrics_{run_stamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    gallery_path = OUTPUT_DIR / f"icme2_counterfactual_gallery_{run_stamp}.png"
    plot_cf_gallery(plot_records, gallery_path, title="ICME Experiment 2 — Counterfactual Gallery")

    found = int(cf_summary["counterfactual_found"].sum())
    total = len(cf_summary)
    print(f"\nCounterfactuals found : {found}/{total}")
    print(f"Summary CSV  → {summary_path}")
    print(f"Metrics CSV  → {metrics_path}")
    print(f"Gallery      → {gallery_path}")
    print(f"Per-CF plots → {cf_ts_dir}")
    if found > 0:
        print("\n── Aggregated metrics ──────────────────────────────────")
        for col in [
            "mean_dice_proximity_score", "mean_dice_sparsity_score",
            "mean_dice_plausibility_score", "dice_diversity_avg_dist",
            "dice_diversity_dpp",
        ]:
            if col in metrics_df.columns:
                print(f"  {col:<38} {metrics_df[col].iloc[0]:.4f}")
    print("\n" + "=" * 70)
    print("Experiment ICME2 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
