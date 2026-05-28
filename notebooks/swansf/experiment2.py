"""
Experiment 2 (SWAN-SF): Window/lag/top-% RF sweep + DiCE counterfactuals.

Loads normalized sample CSVs produced by experiment1, builds FFT features across
multiple observation windows and forecast lags, trains independent Random Forest
models, then generates DiCE counterfactuals and computes proximity/sparsity/
plausibility/diversity metrics. All outputs are written under data/swansf/.

Data split: chronological 70/15/15 (train/val/test) sorted by start_time.

Data format: 60 timesteps per file, 12-minute cadence → 720-minute (12-hour) window.
All index/lag/window parameters below are in TIMESTEPS (1 timestep = 12 minutes).

Pipeline parameters (tune at the top of main()):
  EVENT_INDEX           = 60   (row 60 = end of 12-hour window)
  OBSERVATION_WINDOW    = 48   (48 rows = 576 min = 9.6 h; leaves 12-row lag headroom)
  FORECAST_LAGS         = [0, 6, 12]   rows → [0, 72, 144] min before window end
  FFT_WINDOW_SIZES      = [6, 12, 16, 24, 48]  rows → [72, 144, 192, 288, 576] min
  TOP_PERCENTS          = [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import warnings
from pathlib import Path

import dice_ml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import irfft
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# Make experiment15 / experiment16 importable from notebooks/swansf/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment15 import (
    build_fft_feature_names,
    build_fft_features_all,
    compute_all_metrics,
    count_features_for_cumulative_importance,
    extract_observation_window,
    percent_key,
    pick_positive_class,
    plot_features_needed_for_half_importance,
    select_top_percentage_indices,
)
from experiment16 import (
    build_best_model_feature_bank,
    choose_training_examples_for_counterfactuals,
    make_dice_explainer,
    plot_toppercent_performance_by_lag,
    plot_toppercent_performance_by_window,
    reconstruct_observation_from_fft_features,
    run_independent_window_top_percent_experiment,
    save_model_artifacts,
    select_best,
    select_best_per_lag,
    select_best_window_per_lag_toppercent,
)

_DICE_CHANGE_TOL = 1e-9
FEATURE_COLS = ["TOTUSJH", "TOTPOT", "TOTUSJZ"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SWANSFDataset:
    """Loads normalized SWAN-SF CSVs produced by experiment1."""

    def __init__(
        self,
        manifest_path: Path,
        sample_csv_dir: Path,
        feature_cols: list[str] = FEATURE_COLS,
    ) -> None:
        self.feature_cols = feature_cols
        manifest = pd.read_csv(manifest_path)
        manifest["start_dt"] = pd.to_datetime(manifest["start_time"], errors="coerce")
        manifest = manifest.dropna(subset=["start_dt"]).copy()
        manifest = manifest.sort_values("start_dt").reset_index(drop=True)

        # resolve paths to normalized CSVs in sample_csv_files/{FL,NF}/
        manifest["csv_path"] = manifest.apply(
            lambda r: sample_csv_dir / str(r["class_label"]) / f"{r['sample_id']}.csv",
            axis=1,
        )
        manifest = manifest[manifest["csv_path"].apply(lambda p: Path(p).exists())].copy()
        if len(manifest) == 0:
            raise FileNotFoundError(
                f"No normalized sample CSVs found under {sample_csv_dir}. "
                "Run experiment1.py first."
            )

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        manifest["encoded_label"] = le.fit_transform(manifest["class_label"])
        self.classes_ = le.classes_
        self.metadata = manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        row = self.metadata.iloc[idx]
        df = pd.read_csv(row["csv_path"])
        values = pd.to_numeric(df[self.feature_cols].stack(), errors="coerce").unstack()
        values = values.fillna(0.0).to_numpy(dtype=np.float32)
        return values, int(row["encoded_label"])


def get_swansf_splits_with_ids(
    dataset: SWANSFDataset,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[
    tuple[list[np.ndarray], np.ndarray, list[str]],
    tuple[list[np.ndarray], np.ndarray, list[str]],
    tuple[list[np.ndarray], np.ndarray, list[str]],
]:
    """Chronological proportional split sorted by start_time."""
    n = len(dataset)
    n_train = int(math.floor(n * train_frac))
    n_val = int(math.floor(n * val_frac))

    def _collect(indices: range) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
        X, y, ids = [], [], []
        for i in indices:
            try:
                series, label = dataset[i]
            except Exception as exc:
                print(f"Skipping index {i}: {exc}")
                continue
            X.append(series)
            y.append(label)
            ids.append(str(dataset.metadata.iloc[i]["sample_id"]))
        return X, np.asarray(y, dtype=np.int32), ids

    train = _collect(range(0, n_train))
    val = _collect(range(n_train, n_train + n_val))
    test = _collect(range(n_train + n_val, n))

    print(
        f"Chronological split — train={len(train[0])}, "
        f"val={len(val[0])}, test={len(test[0])}"
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Inverse min-max transform (for counterfactual reconstruction in original units)
# ---------------------------------------------------------------------------

def make_inverse_minmax(
    norm_stats: dict[str, dict[str, float]],
    feature_cols: list[str],
) -> callable:
    def _inverse(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr, dtype=np.float64).copy()
        for c, col in enumerate(feature_cols):
            lo = float(norm_stats[col]["min"])
            hi = float(norm_stats[col]["max"])
            out[:, c] = out[:, c] * (hi - lo) + lo
        return out.astype(np.float32)
    return _inverse


# ---------------------------------------------------------------------------
# DiCE metric helpers (self-contained, mirrors experiment19)
# ---------------------------------------------------------------------------

def _fit_dice_metric_stats(X_train_sel: np.ndarray) -> dict[str, np.ndarray]:
    train = np.asarray(X_train_sel, dtype=np.float64)
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0)
    mad = np.where(mad > 0, mad, 1.0)
    weights = np.where(mad > 0, 1.0 / mad, 1.0)
    weights = np.where(np.isfinite(weights), weights, 1.0)
    return {"feature_mad": mad, "feature_weights": weights}


def _dice_weighted_l1(
    x: np.ndarray,
    y: np.ndarray,
    metric_stats: dict[str, np.ndarray],
    average: bool = False,
) -> np.ndarray:
    w = np.asarray(metric_stats["feature_weights"], dtype=np.float64)
    dist = np.sum(np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)) * w, axis=-1)
    return dist / float(w.size) if average and w.size > 0 else dist


def _compute_cf_metrics(
    original_sel: np.ndarray,
    cf_selected: np.ndarray,
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    cf_prediction: int,
    metric_stats: dict[str, np.ndarray],
) -> dict[str, float | int]:
    original = np.asarray(original_sel, dtype=np.float64)
    cf = np.asarray(cf_selected, dtype=np.float64)
    train = np.asarray(X_train_sel, dtype=np.float64)
    train_labels = np.asarray(y_train, dtype=np.int32)

    changed = np.abs(cf - original) > _DICE_CHANGE_TOL
    changed_count = int(np.sum(changed))
    n_features = int(cf.shape[0])
    sparsity_loss = float(changed_count / n_features) if n_features > 0 else np.nan
    proximity_loss = float(_dice_weighted_l1(cf, original, metric_stats, average=True))

    same_mask = train_labels == int(cf_prediction)
    pool = train[same_mask] if np.any(same_mask) else train
    plausibility_dist = np.nan
    nearest_idx, nearest_label = -1, -1
    if pool.size > 0:
        dists = _dice_weighted_l1(pool, cf, metric_stats, average=True)
        best = int(np.argmin(dists))
        plausibility_dist = float(dists[best])
        if np.any(same_mask):
            nearest_idx = int(np.where(same_mask)[0][best])
        else:
            nearest_idx = best
        nearest_label = int(train_labels[nearest_idx])

    return {
        "changed_feature_count": changed_count,
        "selected_feature_count": n_features,
        "changed_feature_fraction": sparsity_loss,
        "dice_proximity_loss": proximity_loss,
        "dice_proximity_score": float(1.0 / (1.0 + proximity_loss)),
        "dice_sparsity_loss": sparsity_loss,
        "dice_sparsity_score": float(1.0 - sparsity_loss) if np.isfinite(sparsity_loss) else np.nan,
        "dice_plausibility_distance": plausibility_dist,
        "dice_plausibility_score": (
            float(1.0 / (1.0 + plausibility_dist)) if np.isfinite(plausibility_dist) else np.nan
        ),
        "dice_nearest_train_index": nearest_idx,
        "dice_nearest_train_label": nearest_label,
    }


def _compute_diversity_metrics(
    cf_vectors: list[np.ndarray],
    metric_stats: dict[str, np.ndarray],
) -> dict[str, float | int]:
    if len(cf_vectors) < 2:
        return {
            "dice_diversity_dpp": 0.0,
            "dice_diversity_avg_dist": 0.0,
            "dice_diversity_pair_count": 0,
            "dice_mean_pairwise_distance": 0.0,
        }
    cfs = np.asarray(cf_vectors, dtype=np.float64)
    n = int(cfs.shape[0])
    kernel = np.zeros((n, n), dtype=np.float64)
    pair_dists: list[float] = []
    pair_sims: list[float] = []
    for i in range(n):
        for j in range(n):
            d = float(_dice_weighted_l1(cfs[i], cfs[j], metric_stats, average=False))
            s = 1.0 / (1.0 + d)
            kernel[i, j] = s + (0.0001 if i == j else 0.0)
            if i < j:
                pair_dists.append(d)
                pair_sims.append(s)
    avg_sim = float(np.mean(pair_sims)) if pair_sims else 1.0
    return {
        "dice_diversity_dpp": float(np.linalg.det(kernel)),
        "dice_diversity_avg_dist": float(1.0 - avg_sim),
        "dice_diversity_pair_count": len(pair_dists),
        "dice_mean_pairwise_distance": float(np.mean(pair_dists)) if pair_dists else 0.0,
    }


def summarize_counterfactual_quality(cf_df: pd.DataFrame) -> pd.DataFrame:
    if "counterfactual_found" in cf_df.columns:
        found = cf_df[cf_df["counterfactual_found"].astype(bool)].copy()
    else:
        found = cf_df.iloc[0:0].copy()

    summary: dict[str, float | int] = {
        "generated_counterfactual_count": int(len(found)),
        "attempted_counterfactual_count": int(len(cf_df)),
        "counterfactual_success_rate": (
            float(len(found) / len(cf_df)) if len(cf_df) > 0 else np.nan
        ),
    }
    metric_cols = [
        "dice_proximity_loss", "dice_proximity_score",
        "dice_sparsity_loss", "dice_sparsity_score",
        "dice_plausibility_distance", "dice_plausibility_score",
        "changed_feature_count", "changed_feature_fraction",
        "counterfactual_reconstruction_mse", "counterfactual_reconstruction_mae",
        "dice_diversity_dpp", "dice_diversity_avg_dist",
        "dice_diversity_pair_count", "dice_mean_pairwise_distance",
        "num_cfs_generated",
    ]
    for col in metric_cols:
        vals = pd.to_numeric(found[col], errors="coerce") if col in found else pd.Series(dtype=float)
        summary[f"mean_{col}"] = float(vals.mean()) if len(vals.dropna()) > 0 else np.nan
        summary[f"median_{col}"] = float(vals.median()) if len(vals.dropna()) > 0 else np.nan

    return pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Counterfactual generation
# ---------------------------------------------------------------------------

def generate_counterfactuals(
    artifact: dict,
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_examples: list[np.ndarray],
    y_examples: np.ndarray,
    example_ids: list[str],
    classes: np.ndarray,
    channel_names: list[str],
    event_index: int,
    observation_window_size: int,
    output_dir: Path,
    run_stamp: str,
    inverse_transform_fn: callable,
    num_per_class: int = 3,
    num_cfs_per_instance: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[dict]]:
    train_bank = build_best_model_feature_bank(artifact, X_train, event_index, observation_window_size)
    example_bank = build_best_model_feature_bank(artifact, X_examples, event_index, observation_window_size)

    X_train_flat = np.asarray(train_bank["X_train_flat"], dtype=np.float32)
    X_ex_flat = np.asarray(example_bank["X_train_flat"], dtype=np.float32)
    sel_idx = np.asarray(artifact["selected_feature_indices"], dtype=np.int64)
    X_train_sel = X_train_flat[:, sel_idx]
    X_ex_sel = X_ex_flat[:, sel_idx]
    feature_names = list(artifact["selected_feature_names"])
    model = artifact["model"]

    channels = int(train_bank["channels"])
    n_slices = int(train_bank["n_slices"])
    max_coeffs = int(train_bank["max_coeffs"])
    fft_window_size = int(artifact["window_size"])
    lag = int(artifact["forecast_lag_min"])

    metric_stats = _fit_dice_metric_stats(X_train_sel)
    chosen = choose_training_examples_for_counterfactuals(
        X_examples_sel=X_ex_sel,
        y_examples=y_examples,
        example_ids=example_ids,
        model=model,
        classes=classes,
        num_per_class=num_per_class,
    )
    dice = make_dice_explainer(X_train_sel, y_train, feature_names, model)
    permitted_range = {name: [0.0, 1.0] for name in feature_names}

    rows: list[dict] = []
    plot_records: list[dict] = []
    _avg_keys = {
        "changed_feature_count", "changed_feature_fraction",
        "dice_proximity_loss", "dice_proximity_score",
        "dice_sparsity_loss", "dice_sparsity_score",
        "dice_plausibility_distance", "dice_plausibility_score",
    }
    empty_metrics: dict = {
        "changed_feature_count": 0,
        "selected_feature_count": len(feature_names),
        "changed_feature_fraction": np.nan,
        "dice_proximity_loss": np.nan, "dice_proximity_score": np.nan,
        "dice_sparsity_loss": np.nan, "dice_sparsity_score": np.nan,
        "dice_plausibility_distance": np.nan, "dice_plausibility_score": np.nan,
        "dice_nearest_train_index": -1, "dice_nearest_train_label": -1,
    }
    empty_diversity: dict = {
        "dice_diversity_dpp": np.nan, "dice_diversity_avg_dist": np.nan,
        "dice_diversity_pair_count": 0, "dice_mean_pairwise_distance": np.nan,
    }

    for ex in chosen:
        idx = int(ex["example_index"])
        sample_id = str(ex["sample_id"])
        label = int(ex["label"])
        label_name = str(ex["label_name"])
        target_label = 1 - label
        target_label_name = str(classes[target_label])

        original_full = X_ex_flat[idx].astype(np.float64).copy()
        original_sel = X_ex_sel[idx].astype(np.float64).copy()
        query_df = pd.DataFrame([original_sel], columns=feature_names)

        cf_generated = False
        cf_prediction = int(ex["predicted_label"])
        error_msg = ""
        cf_mse = np.nan
        cf_mae = np.nan
        cf_metrics = dict(empty_metrics)
        inst_diversity = dict(empty_diversity)
        num_cfs_generated = 0

        try:
            cf_result = dice.generate_counterfactuals(
                query_df, total_CFs=num_cfs_per_instance, desired_class="opposite",
                features_to_vary="all", permitted_range=permitted_range,
                verbose=False, sample_size=2000, random_seed=random_state + idx,
            )
            cf_df_rows = cf_result.cf_examples_list[0].final_cfs_df
            if len(cf_df_rows) == 0:
                raise RuntimeError("DiCE returned no rows.")

            original_obs = extract_observation_window(
                X_examples[idx], event_index=event_index,
                observation_window_size=observation_window_size, lag_minute=lag,
            )

            inst_cf_sels: list[np.ndarray] = []
            inst_cf_metrics_list: list[dict] = []
            inst_cf_mse_list: list[float] = []
            inst_cf_mae_list: list[float] = []

            for i in range(len(cf_df_rows)):
                cf_sel_i = cf_df_rows.iloc[i][feature_names].astype(float).to_numpy(dtype=np.float64)
                cf_full_i = original_full.copy()
                cf_full_i[sel_idx] = cf_sel_i
                cf_recon_i = reconstruct_observation_from_fft_features(
                    cf_full_i, channels=channels, n_slices=n_slices,
                    max_coeffs=max_coeffs, fft_window_size=fft_window_size,
                )
                cf_pred_i = int(model.predict(cf_sel_i.reshape(1, -1))[0])
                cf_metrics_i = _compute_cf_metrics(
                    original_sel=original_sel, cf_selected=cf_sel_i,
                    X_train_sel=X_train_sel, y_train=y_train,
                    cf_prediction=cf_pred_i, metric_stats=metric_stats,
                )
                inst_cf_sels.append(cf_sel_i)
                inst_cf_metrics_list.append(cf_metrics_i)
                inst_cf_mse_list.append(float(np.mean((original_obs - cf_recon_i) ** 2)))
                inst_cf_mae_list.append(float(np.mean(np.abs(original_obs - cf_recon_i))))

            num_cfs_generated = len(inst_cf_sels)
            best_i = int(np.argmin([m["dice_proximity_loss"] for m in inst_cf_metrics_list]))
            cf_prediction = int(model.predict(inst_cf_sels[best_i].reshape(1, -1))[0])

            cf_metrics = {
                k: float(np.mean([m[k] for m in inst_cf_metrics_list])) if k in _avg_keys
                   else inst_cf_metrics_list[best_i][k]
                for k in inst_cf_metrics_list[0]
            }
            cf_mse = float(np.mean(inst_cf_mse_list))
            cf_mae = float(np.mean(inst_cf_mae_list))
            inst_diversity = _compute_diversity_metrics(inst_cf_sels, metric_stats)

            cf_full_best = original_full.copy()
            cf_full_best[sel_idx] = inst_cf_sels[best_i]
            cf_recon_best = reconstruct_observation_from_fft_features(
                cf_full_best, channels=channels, n_slices=n_slices,
                max_coeffs=max_coeffs, fft_window_size=fft_window_size,
            )
            best_mse = float(np.mean((original_obs - cf_recon_best) ** 2))
            plot_records.append({
                "sample_id": sample_id,
                "label_name": label_name,
                "target_label_name": target_label_name,
                "original_obs": original_obs,
                "cf_recon": cf_recon_best,
                "lag": lag,
                "window_size": fft_window_size,
                "top_percent_key": int(artifact["top_percent_key"]),
                "counterfactual_mse": best_mse,
                "channel_names": channel_names,
            })
            _plot_cf_timeseries(
                sample_id=sample_id, label_name=label_name,
                target_label_name=target_label_name, channel_names=channel_names,
                original_obs=original_obs, cf_recon=cf_recon_best,
                lag=lag, window_size=fft_window_size,
                top_percent_key=int(artifact["top_percent_key"]),
                counterfactual_mse=best_mse,
                output_path=output_dir / "timeseries" / (
                    f"swansf_cf_{sample_id}_lag{lag}_w{fft_window_size}_"
                    f"top{artifact['top_percent_key']}_{label_name}_to_{target_label_name}_{run_stamp}.png"
                ),
            )
            cf_generated = True
        except Exception as exc:
            error_msg = str(exc)

        rows.append({
            "sample_id": sample_id,
            "forecast_lag_min": lag,
            "window_size": fft_window_size,
            "top_percent": float(artifact["top_percent"]),
            "top_percent_label": f"{artifact['top_percent_key']}%",
            "original_label": label,
            "original_label_name": label_name,
            "desired_counterfactual_label": target_label,
            "desired_counterfactual_label_name": target_label_name,
            "counterfactual_predicted_label": cf_prediction,
            "counterfactual_predicted_label_name": str(classes[cf_prediction]),
            "counterfactual_found": bool(cf_generated),
            "num_cfs_requested": num_cfs_per_instance,
            "num_cfs_generated": num_cfs_generated,
            "counterfactual_reconstruction_mse": float(cf_mse) if np.isfinite(cf_mse) else np.nan,
            "counterfactual_reconstruction_mae": float(cf_mae) if np.isfinite(cf_mae) else np.nan,
            **cf_metrics,
            **inst_diversity,
            "error": error_msg,
        })

    return pd.DataFrame(rows), plot_records


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_cf_timeseries(
    sample_id: str,
    label_name: str,
    target_label_name: str,
    channel_names: list[str],
    original_obs: np.ndarray,
    cf_recon: np.ndarray,
    lag: int,
    window_size: int,
    top_percent_key: int,
    counterfactual_mse: float,
    output_path: Path,
) -> None:
    channels = int(original_obs.shape[1])
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(original_obs.shape[0])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for c in range(channels):
        lbl = channel_names[c] if c < len(channel_names) else f"ch{c}"
        color = colors[c % len(colors)]
        ax.plot(x, original_obs[:, c], color=color, linewidth=1.6, label=f"{lbl} original")
        ax.plot(x, cf_recon[:, c], color=color, linewidth=1.6, linestyle="--", label=f"{lbl} cf")
    ax.set_xlabel("Minutes in observation window")
    ax.set_ylabel("Normalized value [0, 1]")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               frameon=False, fontsize=8, ncol=channels * 2)
    fig.suptitle(
        f"CF | {sample_id} | {label_name} -> {target_label_name} | "
        f"lag={lag}, w={window_size}, top%={top_percent_key} | MSE={counterfactual_mse:.3e}",
        fontsize=11, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_counterfactual_gallery(
    plot_records: list[dict],
    output_path: Path,
    title: str = "SWAN-SF Counterfactual Gallery",
) -> None:
    if not plot_records:
        return
    n_plots = min(len(plot_records), 6)
    ncols, nrows = 2, int(math.ceil(n_plots / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, rec in enumerate(plot_records[:n_plots]):
        ax = axes_arr[i]
        orig = np.asarray(rec["original_obs"], dtype=np.float64)
        cf = np.asarray(rec["cf_recon"], dtype=np.float64)
        chs = list(rec["channel_names"])
        x = np.arange(orig.shape[0])
        for c in range(orig.shape[1]):
            lbl = chs[c] if c < len(chs) else f"ch{c}"
            color = colors[c % len(colors)]
            ax.plot(x, orig[:, c], color=color, linewidth=1.3, alpha=0.4,
                    label=f"{lbl} orig" if i == 0 else None)
            ax.plot(x, cf[:, c], color=color, linewidth=1.7, linestyle="--",
                    label=f"{lbl} cf" if i == 0 else None)
        ax.grid(True, alpha=0.2)
        ax.set_title(
            f"{rec['sample_id']} | {rec['label_name']} → {rec['target_label_name']}\n"
            f"MSE={float(rec['counterfactual_mse']):.3e}",
            fontsize=9,
        )

    for j in range(n_plots, len(axes_arr)):
        axes_arr[j].axis("off")

    handles, lbls = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=min(len(handles), 6), frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=13, y=1.005)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    swansf_dir = project_root / "data" / "swansf"

    # --- locate latest experiment1 output (most recently modified stride directory)
    stride_dirs = sorted(
        swansf_dir.glob("partition4_stride_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not stride_dirs:
        raise FileNotFoundError(
            f"No partition4_stride_* directory found under {swansf_dir}. "
            "Run experiment1.py first."
        )
    data_dir = stride_dirs[0]
    manifest_path = data_dir / "selected_files_manifest.csv"
    sample_csv_dir = data_dir / "sample_csv_files"
    norm_meta_path = data_dir / "normalization_metadata.json"

    print(f"Loading data from: {data_dir}")

    with open(norm_meta_path, encoding="utf-8") as f:
        norm_meta = json.load(f)
    norm_stats = norm_meta["stats"]
    inverse_transform = make_inverse_minmax(norm_stats, FEATURE_COLS)

    dataset = SWANSFDataset(
        manifest_path=manifest_path,
        sample_csv_dir=sample_csv_dir,
        feature_cols=FEATURE_COLS,
    )
    (X_train, y_train, train_ids), (X_val, y_val, _), (X_test, y_test, test_ids) = (
        get_swansf_splits_with_ids(dataset)
    )
    print(f"Classes: {dataset.classes_}")

    # --- pipeline parameters (all in timesteps; 1 timestep = 12 min)
    EVENT_INDEX = 60           # end of 12-hour window
    OBSERVATION_WINDOW = 48    # 9.6 h; leaves 12-row (144 min) headroom for max lag
    FORECAST_LAGS = [0, 6, 12] # 0 / 72 / 144 min before window end
    FFT_WINDOW_SIZES = [6, 12, 16, 24, 48]  # 72/144/192/288/576 min; all divide 48
    TOP_PERCENTS = [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]

    run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_root = swansf_dir / "reports" / "experiment2" / run_stamp
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_root}")

    # --- RF sweep
    print("\nRunning RF sweep (window × lag × top%)...")
    results_df, selection_df, model_artifacts = run_independent_window_top_percent_experiment(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        classes=dataset.classes_,
        channel_names=FEATURE_COLS,
        fft_window_sizes=FFT_WINDOW_SIZES,
        forecast_lags=FORECAST_LAGS,
        top_percents=TOP_PERCENTS,
        event_index=EVENT_INDEX,
        observation_window_size=OBSERVATION_WINDOW,
        random_state=42,
    )

    results_path = output_root / f"swansf_exp2_results_{run_stamp}.csv"
    selection_path = output_root / f"swansf_exp2_selection_{run_stamp}.csv"
    results_df.to_csv(results_path, index=False)
    selection_df.to_csv(selection_path, index=False)

    models_dir = output_root / "saved_models"
    save_model_artifacts(model_artifacts, models_dir, run_stamp=run_stamp)

    # --- performance plots
    plot_toppercent_performance_by_lag(
        results_df,
        output_root / f"swansf_exp2_val_css_by_lag_{run_stamp}.png",
        metric_col="val_css",
    )
    plot_toppercent_performance_by_lag(
        results_df,
        output_root / f"swansf_exp2_test_css_by_lag_{run_stamp}.png",
        metric_col="test_css",
    )
    plot_toppercent_performance_by_window(
        results_df,
        output_root / f"swansf_exp2_val_css_by_window_{run_stamp}.png",
        metric_col="val_css",
    )
    plot_toppercent_performance_by_window(
        results_df,
        output_root / f"swansf_exp2_test_css_by_window_{run_stamp}.png",
        metric_col="test_css",
    )
    best_by_lag_top_df = select_best_window_per_lag_toppercent(results_df)
    plot_features_needed_for_half_importance(
        best_by_lag_top_df,
        output_root / f"swansf_exp2_features_cumimp_gt50_{run_stamp}.png",
    )

    # --- best model summary
    best_row = select_best(results_df)
    best_summary_path = output_root / f"swansf_exp2_best_model_summary_{run_stamp}.csv"
    pd.DataFrame([best_row]).to_csv(best_summary_path, index=False)
    print(
        f"\nBest model: lag={int(best_row['forecast_lag_min'])}, "
        f"window={int(best_row['window_size'])}, "
        f"top%={int(round(float(best_row['top_percent']) * 100))}, "
        f"val_css={float(best_row['val_css']):.4f}, "
        f"test_css={float(best_row['test_css']):.4f}"
    )

    # --- counterfactuals: run on each best-per-lag model
    print("\nGenerating DiCE counterfactuals...")
    best_per_lag = select_best_per_lag(results_df)
    best_artifacts_by_lag = [
        next(
            a for a in model_artifacts
            if int(a["forecast_lag_min"]) == int(row["forecast_lag_min"])
            and int(a["window_size"]) == int(row["window_size"])
            and int(a["top_percent_key"]) == int(percent_key(float(row["top_percent"])))
        )
        for _, row in best_per_lag.iterrows()
    ]

    cf_dir = output_root / "counterfactual_timeseries"
    all_cf_frames: list[pd.DataFrame] = []
    all_plot_records: list[dict] = []

    for artifact in best_artifacts_by_lag:
        lag = int(artifact["forecast_lag_min"])
        print(f"  Generating CFs for lag={lag}...")
        cf_df, plot_records = generate_counterfactuals(
            artifact=artifact,
            X_train=X_train, y_train=y_train,
            X_examples=X_test, y_examples=y_test,
            example_ids=test_ids,
            classes=dataset.classes_,
            channel_names=FEATURE_COLS,
            event_index=EVENT_INDEX,
            observation_window_size=OBSERVATION_WINDOW,
            output_dir=cf_dir / f"lag{lag}",
            run_stamp=run_stamp,
            inverse_transform_fn=inverse_transform,
            num_per_class=3,
            num_cfs_per_instance=5,
            random_state=42,
        )
        all_cf_frames.append(cf_df)
        all_plot_records.extend(plot_records)

    if all_cf_frames:
        all_cf_df = pd.concat(all_cf_frames, axis=0, ignore_index=True)
    else:
        all_cf_df = pd.DataFrame()

    cf_summary_path = output_root / f"swansf_exp2_counterfactual_summary_{run_stamp}.csv"
    all_cf_df.to_csv(cf_summary_path, index=False)

    cf_metrics_df = summarize_counterfactual_quality(all_cf_df)
    cf_metrics_path = output_root / f"swansf_exp2_counterfactual_metrics_{run_stamp}.csv"
    cf_metrics_df.to_csv(cf_metrics_path, index=False)

    gallery_path = output_root / f"swansf_exp2_counterfactual_gallery_{run_stamp}.png"
    plot_counterfactual_gallery(all_plot_records, gallery_path)

    # --- final summary
    print("\n" + "=" * 80)
    print("Experiment 2 (SWAN-SF) finished.")
    print(f"  Results CSV          : {results_path}")
    print(f"  Best model summary   : {best_summary_path}")
    print(f"  CF summary CSV       : {cf_summary_path}")
    print(f"  CF metrics CSV       : {cf_metrics_path}")
    print(f"  CF gallery           : {gallery_path}")
    print(f"  Saved models dir     : {models_dir}")
    if len(cf_metrics_df) > 0:
        sr = cf_metrics_df.iloc[0].get("counterfactual_success_rate", np.nan)
        prox = cf_metrics_df.iloc[0].get("mean_dice_proximity_score", np.nan)
        spar = cf_metrics_df.iloc[0].get("mean_dice_sparsity_score", np.nan)
        plaus = cf_metrics_df.iloc[0].get("mean_dice_plausibility_score", np.nan)
        print(f"\n  CF success rate      : {sr:.3f}")
        print(f"  Mean proximity score : {prox:.4f}")
        print(f"  Mean sparsity score  : {spar:.4f}")
        print(f"  Mean plausibility    : {plaus:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
