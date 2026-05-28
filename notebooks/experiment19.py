"""
Experiment 19: Box-Cox counterfactual sweep across feature-varying budgets.

This experiment reuses the latest best Box-Cox model from Experiment 17 and compares
counterfactual generation when DiCE is allowed to vary only low-frequency FFT features
versus broader feature sets, to study the impact on reconstruction smoothness.
"""

from __future__ import annotations

import pickle
import warnings
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox

from experiment15 import TimeSeriesDataset, extract_observation_window, percent_key
from experiment16 import (
    get_splits_with_ids,
    plot_features_needed_for_half_importance,
    plot_toppercent_performance_by_lag,
    plot_toppercent_performance_by_window,
    run_independent_window_top_percent_experiment,
    select_best,
    select_best_window_per_lag_toppercent,
)


# Suppress all warnings to match experiment16 behaviour
warnings.filterwarnings("ignore")


_DICE_CHANGE_TOLERANCE = 1e-9

_CHANNEL_MAX = {
    "p3_flux_ic": 43500.0,
    "p5_flux_ic": 2780.0,
    "p7_flux_ic": 652.0,
    "long": 46714.3,
}
_TARGET_PREEXP_MAX = 10.0  # keeps exp well below float32 overflow
_CHANNEL_SCALE = {k: (_TARGET_PREEXP_MAX / v) for k, v in _CHANNEL_MAX.items()}


def _as_float32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def build_channel_scale(target_preexp_max: float) -> dict[str, float]:
    return {k: (target_preexp_max / v) for k, v in _CHANNEL_MAX.items()}


def fit_exp_clip_max(
    X_train: list[np.ndarray],
    channel_order: list[str],
    channel_scale: dict[str, float],
) -> float:
    flat = np.concatenate([np.asarray(x, dtype=np.float64) for x in X_train], axis=0)
    scaled = flat.copy()
    for idx, ch in enumerate(channel_order):
        scale = channel_scale.get(ch, 1.0)
        scaled[:, idx] *= scale
    return float(np.max(scaled))


def safe_exp_transform(
    sample: np.ndarray,
    channel_order: list[str],
    clip_max: float,
    channel_scale: dict[str, float],
) -> np.ndarray:
    """Scale per-channel then exponentiate for model-space features."""
    arr = np.asarray(sample, dtype=np.float64).copy()
    for idx, ch in enumerate(channel_order):
        scale = channel_scale.get(ch, 1.0)
        arr[:, idx] *= scale
    arr = np.clip(arr, None, clip_max)
    return np.exp(arr, dtype=np.float64).astype(np.float32)


def inverse_exp(sample: np.ndarray, channel_order: list[str], channel_scale: dict[str, float]) -> np.ndarray:
    arr = np.log(np.clip(np.asarray(sample, dtype=np.float64), 1e-12, None))
    for idx, ch in enumerate(channel_order):
        scale = channel_scale.get(ch, 1.0)
        arr[:, idx] /= scale
    return arr.astype(np.float32)


def log1p_transform(sample: np.ndarray) -> np.ndarray:
    """Log-compress positive flux values for model-space features."""
    return np.log1p(np.clip(np.asarray(sample, dtype=np.float64), 0.0, None)).astype(np.float32)


def inverse_log1p(sample: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(sample, dtype=np.float64)).astype(np.float32)


def fit_box_cox_transform(X_train: list[np.ndarray]) -> tuple[float, float, callable, callable]:
    """Fit a Box-Cox transform on the training split for model-space features."""
    flat = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in X_train], axis=0)
    min_val = float(np.min(flat))
    shift = 1e-6 if min_val > 0 else (-min_val + 1e-6)
    shifted = flat + shift
    _, lambda_ = stats.boxcox(shifted)

    def forward(sample: np.ndarray) -> np.ndarray:
        shape = np.asarray(sample).shape
        arr = np.asarray(sample, dtype=np.float64).reshape(-1) + shift
        return stats.boxcox(arr, lmbda=lambda_).reshape(shape).astype(np.float32)

    def inverse(sample: np.ndarray) -> np.ndarray:
        shape = np.asarray(sample).shape
        restored = inv_boxcox(np.asarray(sample, dtype=np.float64).reshape(-1), lambda_) - shift
        return restored.reshape(shape).astype(np.float32)

    return shift, lambda_, forward, inverse


def transform_dataset(X_data: list[np.ndarray], transform_fn) -> list[np.ndarray]:
    return [transform_fn(x) for x in X_data]


def feature_coeff_index(feature_name: str) -> int | None:
    name = str(feature_name)
    marker = "_coeff"
    pos = name.rfind(marker)
    if pos < 0:
        return None
    token = name[pos + len(marker) :]
    return int(token) if token.isdigit() else None


def low_frequency_feature_names(feature_names: list[str], max_coeff_index: int) -> list[str]:
    """Keep only low-frequency FFT features up to the requested coefficient index."""
    kept: list[str] = []
    for name in feature_names:
        coeff_idx = feature_coeff_index(name)
        if coeff_idx is None:
            continue
        if coeff_idx <= int(max_coeff_index):
            kept.append(str(name))
    return kept


def features_to_vary_for_box_cox(feature_names: list[str], cutoff: str) -> list[str]:
    """Map a box_cox sweep label to the subset of features DiCE may modify."""
    cutoff_key = str(cutoff).lower().strip()
    if cutoff_key in {"0-9", "0_9", "low9"}:
        return low_frequency_feature_names(feature_names, max_coeff_index=9)
    if cutoff_key in {"0-20", "0_20", "low20"}:
        return low_frequency_feature_names(feature_names, max_coeff_index=20)
    if cutoff_key in {"0-all", "all", "full"}:
        return list(feature_names)
    raise ValueError(f"Unknown box_cox feature-vary cutoff '{cutoff}'.")


def fit_dice_metric_stats(X_train_sel: np.ndarray) -> dict[str, np.ndarray]:
    """Precompute inverse-MAD weights used by DiCE for continuous features."""
    train = np.asarray(X_train_sel, dtype=np.float64)
    feature_median = np.median(train, axis=0)
    feature_mad = np.median(np.abs(train - feature_median), axis=0)
    feature_mad = np.where(feature_mad > 0, feature_mad, 1.0)

    feature_weights = np.divide(
        1.0,
        feature_mad,
        out=np.ones_like(feature_mad, dtype=np.float64),
        where=feature_mad > 0,
    )
    feature_weights = np.where(np.isfinite(feature_weights), feature_weights, 1.0)

    return {
        "feature_mad": feature_mad,
        "feature_weights": feature_weights,
    }


def dice_weighted_l1_distance(
    x: np.ndarray,
    y: np.ndarray,
    metric_stats: dict[str, np.ndarray],
    average_over_features: bool = False,
) -> np.ndarray:
    """DiCE-style continuous distance: sum_j abs(x_j - y_j) / MAD_j."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    feature_weights = np.asarray(metric_stats["feature_weights"], dtype=np.float64)

    weighted = np.abs(x_arr - y_arr) * feature_weights
    distances = np.sum(weighted, axis=-1)
    if average_over_features and feature_weights.size > 0:
        distances = distances / int(feature_weights.size)
    return distances


def compute_dice_counterfactual_metrics(
    original_sel: np.ndarray,
    cf_selected: np.ndarray,
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    cf_prediction: int,
    metric_stats: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Compute per-CF metrics in the same selected FFT feature space DiCE edits."""
    original = np.asarray(original_sel, dtype=np.float64)
    cf = np.asarray(cf_selected, dtype=np.float64)
    train = np.asarray(X_train_sel, dtype=np.float64)
    train_labels = np.asarray(y_train, dtype=np.int32)

    changed_mask = np.abs(cf - original) > _DICE_CHANGE_TOLERANCE
    changed_feature_count = int(np.sum(changed_mask))
    feature_count = int(cf.shape[0])
    sparsity_loss = float(changed_feature_count / feature_count) if feature_count > 0 else np.nan

    proximity_loss = float(
        dice_weighted_l1_distance(
            cf,
            original,
            metric_stats=metric_stats,
            average_over_features=True,
        )
    )

    same_class_mask = train_labels == int(cf_prediction)
    plausibility_pool = train[same_class_mask] if np.any(same_class_mask) else train
    nearest_train_index = -1
    nearest_train_label = -1
    plausibility_distance = np.nan
    if plausibility_pool.size > 0:
        distances = dice_weighted_l1_distance(
            plausibility_pool,
            cf,
            metric_stats=metric_stats,
            average_over_features=True,
        )
        nearest_pool_index = int(np.argmin(distances))
        plausibility_distance = float(distances[nearest_pool_index])
        if np.any(same_class_mask):
            original_indices = np.where(same_class_mask)[0]
            nearest_train_index = int(original_indices[nearest_pool_index])
        else:
            nearest_train_index = int(nearest_pool_index)
        nearest_train_label = int(train_labels[nearest_train_index])

    return {
        "changed_feature_count": changed_feature_count,
        "selected_feature_count": feature_count,
        "changed_feature_fraction": sparsity_loss,
        "dice_proximity_loss": proximity_loss,
        "dice_proximity_score": float(1.0 / (1.0 + proximity_loss)),
        "dice_sparsity_loss": sparsity_loss,
        "dice_sparsity_score": float(1.0 - sparsity_loss) if np.isfinite(sparsity_loss) else np.nan,
        "dice_plausibility_distance": plausibility_distance,
        "dice_plausibility_score": (
            float(1.0 / (1.0 + plausibility_distance))
            if np.isfinite(plausibility_distance)
            else np.nan
        ),
        "dice_nearest_train_index": nearest_train_index,
        "dice_nearest_train_label": nearest_train_label,
    }


def compute_dice_diversity_metrics(
    cf_vectors: list[np.ndarray],
    metric_stats: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Compute DiCE DPP and avg-distance diversity over the accepted CF set."""
    if len(cf_vectors) < 2:
        return {
            "dice_diversity_dpp": 0.0,
            "dice_diversity_avg_dist": 0.0,
            "dice_diversity_pair_count": 0,
            "dice_mean_pairwise_distance": 0.0,
        }

    cfs = np.asarray(cf_vectors, dtype=np.float64)
    n_cfs = int(cfs.shape[0])
    kernel = np.zeros((n_cfs, n_cfs), dtype=np.float64)
    pairwise_distances: list[float] = []
    pairwise_similarities: list[float] = []

    for i in range(n_cfs):
        for j in range(n_cfs):
            distance = float(
                dice_weighted_l1_distance(
                    cfs[i],
                    cfs[j],
                    metric_stats=metric_stats,
                    average_over_features=False,
                )
            )
            similarity = float(1.0 / (1.0 + distance))
            kernel[i, j] = similarity
            if i == j:
                kernel[i, j] += 0.0001
            elif i < j:
                pairwise_distances.append(distance)
                pairwise_similarities.append(similarity)

    pair_count = len(pairwise_distances)
    avg_similarity = float(np.mean(pairwise_similarities)) if pairwise_similarities else 1.0
    return {
        "dice_diversity_dpp": float(np.linalg.det(kernel)),
        "dice_diversity_avg_dist": float(1.0 - avg_similarity),
        "dice_diversity_pair_count": int(pair_count),
        "dice_mean_pairwise_distance": float(np.mean(pairwise_distances)) if pairwise_distances else 0.0,
    }


def summarize_counterfactual_quality(counterfactual_df: pd.DataFrame) -> pd.DataFrame:
    if "counterfactual_found" in counterfactual_df:
        found_df = counterfactual_df[counterfactual_df["counterfactual_found"].astype(bool)].copy()
    else:
        found_df = counterfactual_df.iloc[0:0].copy()
    summary: dict[str, float | int | str] = {
        "generated_counterfactual_count": int(len(found_df)),
        "attempted_counterfactual_count": int(len(counterfactual_df)),
        "counterfactual_success_rate": (
            float(len(found_df) / len(counterfactual_df)) if len(counterfactual_df) > 0 else np.nan
        ),
    }

    metric_cols = [
        "dice_proximity_loss",
        "dice_proximity_score",
        "dice_sparsity_loss",
        "dice_sparsity_score",
        "dice_plausibility_distance",
        "dice_plausibility_score",
        "changed_feature_count",
        "changed_feature_fraction",
        "counterfactual_reconstruction_mse",
        "counterfactual_reconstruction_mae",
    ]
    for col in metric_cols:
        values = pd.to_numeric(found_df[col], errors="coerce") if col in found_df else pd.Series(dtype=float)
        summary[f"mean_{col}"] = float(values.mean()) if len(values.dropna()) > 0 else np.nan
        summary[f"median_{col}"] = float(values.median()) if len(values.dropna()) > 0 else np.nan

    diversity_cols = [
        "dice_diversity_dpp",
        "dice_diversity_avg_dist",
        "dice_diversity_pair_count",
        "dice_mean_pairwise_distance",
    ]
    for col in diversity_cols:
        values = pd.to_numeric(found_df[col], errors="coerce") if col in found_df else pd.Series(dtype=float)
        summary[col] = float(values.dropna().iloc[0]) if len(values.dropna()) > 0 else np.nan

    return pd.DataFrame([summary])


def _extract_run_stamp_from_stem(stem: str, prefix: str) -> str | None:
    if not stem.startswith(prefix + "_"):
        return None
    return stem[len(prefix) + 1 :]


def load_latest_experiment17_box_cox_artifact(project_root: Path) -> tuple[dict, pd.Series, str]:
    box_dir = project_root / "data" / "reports" / "experiment17" / "box_cox"
    summary_paths = sorted(box_dir.glob("experiment17_box_cox_best_model_summary_*.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"No Experiment 17 box_cox best-model summaries found in {box_dir}.")

    def sort_key(path: Path) -> str:
        stamp = _extract_run_stamp_from_stem(path.stem, "experiment17_box_cox_best_model_summary")
        return stamp or ""

    latest_summary = max(summary_paths, key=sort_key)
    run_stamp = _extract_run_stamp_from_stem(latest_summary.stem, "experiment17_box_cox_best_model_summary")
    if not run_stamp:
        raise ValueError(f"Could not parse run stamp from '{latest_summary.name}'.")

    best_row = pd.read_csv(latest_summary).iloc[0]
    model_path = (
        box_dir
        / "saved_models"
        / f"experiment17_box_cox_rf_w{int(best_row['window_size'])}_lag{int(best_row['forecast_lag_min'])}_top{int(percent_key(float(best_row['top_percent'])))}_{run_stamp}.pkl"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Missing Experiment 17 box_cox model artifact: {model_path}")

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    return artifact, best_row, run_stamp


def box_cox_provenance_fields(run_stamp: str, best_row: pd.Series) -> dict[str, str]:
    model_key = f"experiment17_box_cox_rf_w{int(best_row['window_size'])}_lag{int(best_row['forecast_lag_min'])}_top{int(percent_key(float(best_row['top_percent'])))}_{run_stamp}_box_cox.pkl"
    summary_key = f"experiment17_box_cox_best_model_summary_{run_stamp}.csv"
    return {
        "source_experiment": "experiment17",
        "source_method": "box_cox",
        "source_run_stamp": run_stamp,
        "source_summary_path": str(Path("data") / "reports" / "experiment17" / "box_cox" / summary_key),
        "source_model_path": str(Path("data") / "reports" / "experiment17" / "box_cox" / "saved_models" / model_key),
    }


def save_model_artifacts_v17(model_artifacts: list[dict], output_dir: Path, run_stamp: str, prefix: str) -> list[Path]:
    """Same as experiment16 saver but with a configurable filename prefix."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for artifact in model_artifacts:
        lag = int(artifact["forecast_lag_min"])
        window = int(artifact["window_size"])
        top_key = int(artifact["top_percent_key"])
        path = output_dir / f"{prefix}_rf_w{window}_lag{lag}_top{top_key}_{run_stamp}.pkl"
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        saved_paths.append(path)

    return saved_paths


def run_variant(
    variant_name: str,
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_val: list[np.ndarray],
    y_val: np.ndarray,
    X_test: list[np.ndarray],
    y_test: np.ndarray,
    test_ids: list[str],
    classes: np.ndarray,
    channel_names: list[str],
    event_onset_index: int,
    observation_window_size: int,
    fft_window_sizes: list[int],
    forecast_lags: list[int],
    top_percents: list[float],
    output_root: Path,
    run_stamp: str,
) -> dict:
    """Run one pipeline variant (baseline or exp_fft) and return key artifacts."""

    variant_stamp = f"{run_stamp}_{variant_name}"
    out_dir = output_root / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df, selection_df, model_artifacts = run_independent_window_top_percent_experiment(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=classes,
        channel_names=channel_names,
        fft_window_sizes=fft_window_sizes,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        random_state=42,
    )

    results_path = out_dir / f"experiment19_{variant_name}_results_{variant_stamp}.csv"
    selection_path = out_dir / f"experiment19_{variant_name}_selection_{variant_stamp}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    selection_df.to_csv(selection_path, index=False)

    models_dir = out_dir / "saved_models"
    saved_model_paths = save_model_artifacts_v17(
        model_artifacts=model_artifacts,
        output_dir=models_dir,
        run_stamp=variant_stamp,
        prefix=f"experiment19_{variant_name}",
    )

    val_plot_by_lag_path = out_dir / f"experiment19_{variant_name}_val_css_by_lag_{variant_stamp}.png"
    plot_toppercent_performance_by_lag(results_df, val_plot_by_lag_path, metric_col="val_css")

    test_plot_by_lag_path = out_dir / f"experiment19_{variant_name}_test_css_by_lag_{variant_stamp}.png"
    plot_toppercent_performance_by_lag(results_df, test_plot_by_lag_path, metric_col="test_css")

    val_plot_by_window_path = out_dir / f"experiment19_{variant_name}_val_css_by_window_{variant_stamp}.png"
    plot_toppercent_performance_by_window(results_df, val_plot_by_window_path, metric_col="val_css")

    test_plot_by_window_path = out_dir / f"experiment19_{variant_name}_test_css_by_window_{variant_stamp}.png"
    plot_toppercent_performance_by_window(results_df, test_plot_by_window_path, metric_col="test_css")

    feature_count_plot_path = out_dir / f"experiment19_{variant_name}_features_required_cumimp_gt50_{variant_stamp}.png"
    best_by_lag_toppercent_df = select_best_window_per_lag_toppercent(results_df)
    plot_features_needed_for_half_importance(best_by_lag_toppercent_df, feature_count_plot_path)

    best_row = select_best(results_df)
    best_artifact = next(
        artifact
        for artifact in model_artifacts
        if int(artifact["forecast_lag_min"]) == int(best_row["forecast_lag_min"])
        and int(artifact["window_size"]) == int(best_row["window_size"])
        and int(artifact["top_percent_key"]) == int(percent_key(float(best_row["top_percent"])))
    )
    best_summary_path = out_dir / f"experiment19_{variant_name}_best_model_summary_{variant_stamp}.csv"
    best_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([best_row]).to_csv(best_summary_path, index=False)

    return {
        "variant": variant_name,
        "results_path": results_path,
        "selection_path": selection_path,
        "best_summary_path": best_summary_path,
        "models_dir": models_dir,
        "saved_model_paths": saved_model_paths,
        "results_df": results_df,
        "best_row": best_row,
    }


def generate_transform_counterfactual_reports(
    artifact: dict,
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_examples_raw: list[np.ndarray],
    X_examples: list[np.ndarray],
    y_examples: np.ndarray,
    example_ids: list[str],
    classes: np.ndarray,
    channel_names: list[str],
    event_index: int,
    observation_window_size: int,
    output_dir: Path,
    run_stamp: str,
    inverse_transform_fn,
    method_name: str,
    features_to_vary="all",
    features_to_vary_label: str | None = None,
    num_per_class: int = 3,
    num_cfs_per_instance: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    from experiment16 import (
        build_best_model_feature_bank,
        choose_training_examples_for_counterfactuals,
        make_dice_explainer,
        plot_counterfactual_channel_timeseries,
        plot_counterfactual_timeseries,
        reconstruct_observation_from_fft_features,
    )

    train_feature_bank = build_best_model_feature_bank(
        artifact=artifact,
        X_data=X_train,
        event_index=event_index,
        observation_window_size=observation_window_size,
    )
    example_feature_bank = build_best_model_feature_bank(
        artifact=artifact,
        X_data=X_examples,
        event_index=event_index,
        observation_window_size=observation_window_size,
    )
    X_train_flat = np.asarray(train_feature_bank["X_train_flat"], dtype=np.float32)
    X_examples_flat = np.asarray(example_feature_bank["X_train_flat"], dtype=np.float32)
    selected_idx = np.asarray(artifact["selected_feature_indices"], dtype=np.int64)
    X_train_sel = X_train_flat[:, selected_idx]
    X_examples_sel = X_examples_flat[:, selected_idx]
    feature_names = list(artifact["selected_feature_names"])
    model = artifact["model"]
    metric_stats = fit_dice_metric_stats(X_train_sel)

    chosen_examples = choose_training_examples_for_counterfactuals(
        X_examples_sel=X_examples_sel,
        y_examples=y_examples,
        example_ids=example_ids,
        model=model,
        classes=classes,
        num_per_class=num_per_class,
        chosen_sample_ids=None,
    )

    dice = make_dice_explainer(
        X_train_sel=X_train_sel,
        y_train=y_train,
        feature_names=feature_names,
        model=model,
    )

    features_to_vary_count = (
        len(feature_names)
        if features_to_vary == "all"
        else len([feature for feature in features_to_vary if feature in feature_names])
    )
    rows: list[dict[str, object]] = []
    plot_records: list[dict[str, object]] = []
    channels = int(train_feature_bank["channels"])
    n_slices = int(train_feature_bank["n_slices"])
    max_coeffs = int(train_feature_bank["max_coeffs"])
    fft_window_size = int(artifact["window_size"])
    lag = int(artifact["forecast_lag_min"])
    method_dir = output_dir / method_name

    _scalar_metric_keys = [
        "changed_feature_count", "selected_feature_count", "changed_feature_fraction",
        "dice_proximity_loss", "dice_proximity_score", "dice_sparsity_loss",
        "dice_sparsity_score", "dice_plausibility_distance", "dice_plausibility_score",
    ]
    _index_metric_keys = ["dice_nearest_train_index", "dice_nearest_train_label"]

    for example in chosen_examples:
        idx = int(example["example_index"])
        sample_id = str(example["sample_id"])
        label = int(example["label"])
        label_name = str(example["label_name"])
        original_full = X_examples_flat[idx].astype(np.float64).copy()
        original_sel = X_examples_sel[idx].astype(np.float64).copy()
        query_df = pd.DataFrame([original_sel], columns=feature_names)

        cf_prediction = int(example["predicted_label"])
        cf_generated = False
        error_message = ""
        counterfactual_mse = np.nan
        counterfactual_mae = np.nan
        metric_values: dict[str, float | int] = {
            "changed_feature_count": 0,
            "selected_feature_count": int(len(feature_names)),
            "changed_feature_fraction": np.nan,
            "dice_proximity_loss": np.nan,
            "dice_proximity_score": np.nan,
            "dice_sparsity_loss": np.nan,
            "dice_sparsity_score": np.nan,
            "dice_plausibility_distance": np.nan,
            "dice_plausibility_score": np.nan,
            "dice_nearest_train_index": -1,
            "dice_nearest_train_label": -1,
        }
        instance_diversity: dict[str, float | int] = {
            "dice_diversity_dpp": np.nan,
            "dice_diversity_avg_dist": np.nan,
            "dice_diversity_pair_count": 0,
            "dice_mean_pairwise_distance": np.nan,
        }
        target_label = 1 - int(label)
        target_label_name = str(classes[int(target_label)])

        try:
            cf_result = dice.generate_counterfactuals(
                query_df,
                total_CFs=num_cfs_per_instance,
                desired_class="opposite",
                features_to_vary=features_to_vary,
                verbose=False,
                sample_size=2000,
                random_seed=random_state + idx,
            )
            cf_df = cf_result.cf_examples_list[0].final_cfs_df
            if len(cf_df) == 0:
                raise RuntimeError("DiCE returned no counterfactual rows.")

            original_obs = extract_observation_window(
                X_examples_raw[idx],
                event_index=event_index,
                observation_window_size=observation_window_size,
                lag_minute=lag,
            )

            instance_cf_vectors: list[np.ndarray] = []
            per_cf_metrics: list[dict] = []
            per_cf_mse: list[float] = []
            per_cf_mae: list[float] = []
            first_cf_recon: np.ndarray | None = None

            for cf_row_idx, (_, cf_row) in enumerate(cf_df.iterrows()):
                cf_selected_i = cf_row[feature_names].astype(float).to_numpy(dtype=np.float64)
                cf_pred_i = int(model.predict(cf_selected_i.reshape(1, -1))[0])
                cf_full_i = original_full.copy()
                cf_full_i[selected_idx] = cf_selected_i
                cf_recon_i = inverse_transform_fn(
                    reconstruct_observation_from_fft_features(
                        cf_full_i,
                        channels=channels,
                        n_slices=n_slices,
                        max_coeffs=max_coeffs,
                        fft_window_size=fft_window_size,
                    )
                )
                mse_i = float(np.mean((original_obs - cf_recon_i) ** 2))
                mae_i = float(np.mean(np.abs(original_obs - cf_recon_i)))
                m_i = compute_dice_counterfactual_metrics(
                    original_sel=original_sel,
                    cf_selected=cf_selected_i,
                    X_train_sel=X_train_sel,
                    y_train=y_train,
                    cf_prediction=cf_pred_i,
                    metric_stats=metric_stats,
                )
                instance_cf_vectors.append(cf_selected_i)
                per_cf_metrics.append(m_i)
                per_cf_mse.append(mse_i)
                per_cf_mae.append(mae_i)
                if cf_row_idx == 0:
                    first_cf_recon = cf_recon_i
                    cf_prediction = cf_pred_i

            counterfactual_mse = float(np.mean(per_cf_mse))
            counterfactual_mae = float(np.mean(per_cf_mae))

            metric_values = {
                k: float(np.nanmean([m[k] for m in per_cf_metrics]))
                for k in _scalar_metric_keys
            }
            metric_values.update({k: per_cf_metrics[0][k] for k in _index_metric_keys})

            instance_diversity = compute_dice_diversity_metrics(instance_cf_vectors, metric_stats)

            plot_records.append(
                {
                    "sample_id": sample_id,
                    "label_name": label_name,
                    "target_label_name": target_label_name,
                    "original_obs": original_obs,
                    "cf_recon": first_cf_recon,
                    "lag": lag,
                    "window_size": fft_window_size,
                    "top_percent_key": int(artifact["top_percent_key"]),
                    "counterfactual_mse": counterfactual_mse,
                    "channel_names": channel_names,
                }
            )

            combined_plot_path = method_dir / "combined" / (
                f"experiment19_{method_name}_cf_sample_{sample_id}_lag{lag}_w{fft_window_size}_"
                f"top{int(artifact['top_percent_key'])}_{label_name}_to_{target_label_name}_{run_stamp}_combined.png"
            )
            plot_counterfactual_timeseries(
                sample_id=sample_id,
                label_name=f"{method_name}: {label_name}",
                target_label_name=target_label_name,
                channel_names=channel_names,
                original_obs=original_obs,
                cf_recon=first_cf_recon,
                lag=lag,
                window_size=fft_window_size,
                top_percent_key=int(artifact["top_percent_key"]),
                counterfactual_mse=counterfactual_mse,
                output_path=combined_plot_path,
            )
            log_combined_plot_path = method_dir / "counterfactual_timeseries" / run_stamp / "log_domain" / (
                f"experiment19_{method_name}_cf_sample_{sample_id}_lag{lag}_w{fft_window_size}_"
                f"top{int(artifact['top_percent_key'])}_{label_name}_to_{target_label_name}_{run_stamp}_combined.png"
            )
            plot_counterfactual_log_timeseries(
                sample_id=sample_id,
                label_name=f"{method_name}: {label_name}",
                target_label_name=target_label_name,
                channel_names=channel_names,
                original_obs=original_obs,
                cf_recon=first_cf_recon,
                lag=lag,
                window_size=fft_window_size,
                top_percent_key=int(artifact["top_percent_key"]),
                counterfactual_mse=counterfactual_mse,
                output_path=log_combined_plot_path,
            )
            cf_generated = True
        except Exception as exc:
            error_message = str(exc)
            if cf_prediction < 0 or cf_prediction >= len(classes):
                cf_prediction = int(example["predicted_label"])

        rows.append(
            {
                "method": method_name,
                "counterfactual_cutoff": str(features_to_vary_label or "all"),
                "sample_id": sample_id,
                "forecast_lag_min": int(lag),
                "window_size": int(fft_window_size),
                "top_percent": float(artifact["top_percent"]),
                "top_percent_label": f"{int(artifact['top_percent_key'])}%",
                "features_to_vary_count": int(features_to_vary_count),
                "features_to_vary_fraction": float(features_to_vary_count / len(feature_names)),
                "original_label": int(label),
                "original_label_name": label_name,
                "desired_counterfactual_label": int(target_label),
                "desired_counterfactual_label_name": target_label_name,
                "counterfactual_predicted_label": int(cf_prediction),
                "counterfactual_predicted_label_name": str(classes[int(cf_prediction)]),
                "counterfactual_found": bool(cf_generated),
                "counterfactual_reconstruction_mse": (
                    float(counterfactual_mse) if np.isfinite(counterfactual_mse) else np.nan
                ),
                "counterfactual_reconstruction_mae": (
                    float(counterfactual_mae) if np.isfinite(counterfactual_mae) else np.nan
                ),
                **metric_values,
                **instance_diversity,
                "error": error_message,
            }
        )

    return pd.DataFrame(rows), plot_records


def plot_counterfactual_gallery(
    method_name: str,
    plot_records: list[dict[str, object]],
    output_path: Path,
) -> None:
    if not plot_records:
        return

    import matplotlib.pyplot as plt

    n_plots = min(len(plot_records), 6)
    ncols = 2
    nrows = int(math.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.6 * nrows), sharex=True)
    axes_arr = np.asarray(axes).reshape(-1)
    colors = ["#FF0000", "#1b5c0c", "#FFA500", "#0000FF"]

    for i, record in enumerate(plot_records[:n_plots]):
        ax = axes_arr[i]
        original_obs = np.asarray(record["original_obs"], dtype=np.float64)
        cf_recon = np.asarray(record["cf_recon"], dtype=np.float64)
        channel_names = list(record["channel_names"])
        x = np.arange(original_obs.shape[0], dtype=np.int64)

        for c in range(original_obs.shape[1]):
            label = channel_names[c] if c < len(channel_names) else f"ch{c}"
            color = colors[c % len(colors)]
            ax.plot(x, original_obs[:, c], color=color, linewidth=1.4, alpha=0.35, label=f"{label} original" if i == 0 else None)
            ax.plot(x, cf_recon[:, c], color=color, linewidth=1.8, linestyle="--", label=f"{label} cf" if i == 0 else None)

        all_values = np.concatenate(
            [
                np.asarray(original_obs, dtype=np.float64).ravel(),
                np.asarray(cf_recon, dtype=np.float64).ravel(),
            ],
            axis=0,
        )
        all_values = all_values[np.isfinite(all_values) & (all_values > 0)]
        if all_values.size > 0:
            ax.set_ylim(bottom=float(np.min(all_values)) * 0.9, top=float(np.max(all_values)) * 2.0)
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{record['sample_id']} | {record['label_name']} -> {record['target_label_name']}\n"
            f"MSE={float(record['counterfactual_mse']):.3e}",
            fontsize=10,
        )

    for j in range(n_plots, len(axes_arr)):
        axes_arr[j].axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Experiment 17 Counterfactual Gallery | {method_name}", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_counterfactual_log_timeseries(
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
    import matplotlib.pyplot as plt

    original_log = np.log10(np.clip(np.asarray(original_obs, dtype=np.float64), 1e-12, None))
    cf_log = np.log10(np.clip(np.asarray(cf_recon, dtype=np.float64), 1e-12, None))

    channels = int(original_log.shape[1])
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    x = np.arange(original_log.shape[0], dtype=np.int64)
    custom_colors = ["#FF0000", "#1b5c0c", "#FFA500", "#0000FF"]
    colors = custom_colors

    for c in range(channels):
        label = channel_names[c] if c < len(channel_names) else f"ch{c}"
        color = colors[c % len(colors)]
        ax.plot(x, original_log[:, c], color=color, linewidth=1.8, label=f"{label} original")
        ax.plot(x, cf_log[:, c], color=color, linewidth=1.8, linestyle="--", label=f"{label} counterfactual")

    ax.set_xlabel("Minutes in observation window")
    ax.set_ylabel("log10(pfu)")
    all_values = np.concatenate([original_log.ravel(), cf_log.ravel()], axis=0)
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size > 0:
        ax.set_ylim(bottom=float(np.min(all_values)) * 0.9, top=float(np.max(all_values)) * 1.1)
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), frameon=False, fontsize=8, ncol=4)
    fig.suptitle(
        f"Counterfactual Time Series | sample={sample_id} | {label_name} -> {target_label_name} | "
        f"best model: lag={int(lag)}, window={int(window_size)}, top%={int(top_percent_key)} | "
        f"cf recon MSE={counterfactual_mse:.3e}",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"

    raw_dir = data_root / "raw"
    labels_file = data_root / "SEP_class_labels.csv"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Missing data directory: '{raw_dir}'. Expected raw CSV files under data/raw/."
        )

    target_channels = ["p3_flux_ic", "p5_flux_ic", "p7_flux_ic", "long"]
    dataset = TimeSeriesDataset(
        data_dir=raw_dir,
        labels_file=labels_file,
        filename_col="File",
        label_col="Label",
        feature_cols=target_channels,
    )

    (X_train, y_train, train_ids), (X_val, y_val, _), (X_test, y_test, test_ids) = get_splits_with_ids(dataset)
    _boxcox_shift, _boxcox_lambda, boxcox_forward, boxcox_inverse = fit_box_cox_transform(X_train)

    output_root = data_root / "reports" / "experiment19"
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    method_results: list[dict[str, object]] = []
    method_name = "box_cox"
    print("Experiment 19:box_cox variant starting...")
    best_artifact, best_row, exp17_run_stamp = load_latest_experiment17_box_cox_artifact(project_root)
    provenance = box_cox_provenance_fields(exp17_run_stamp, best_row)

    X_train_m = transform_dataset(X_train, boxcox_forward)
    X_test_m = transform_dataset(X_test, boxcox_forward)

    for cutoff in ["0-all"]:
        features_to_vary = features_to_vary_for_box_cox(list(best_artifact["selected_feature_names"]), cutoff)
        if not features_to_vary:
            raise RuntimeError(f"No features found for box_cox cutoff '{cutoff}'.")
        cutoff_dir = output_root / method_name / cutoff / "counterfactual_timeseries" / run_stamp
        counterfactual_summary_path = (
            output_root
            / method_name
            / cutoff
            / f"experiment19_{method_name}_{cutoff}_counterfactual_summary_{run_stamp}.csv"
        )
        counterfactual_df, plot_records = generate_transform_counterfactual_reports(
            artifact=best_artifact,
            X_train=X_train_m,
            y_train=y_train,
            X_examples_raw=X_test,
            X_examples=X_test_m,
            y_examples=y_test,
            example_ids=test_ids,
            classes=dataset.classes_,
            channel_names=target_channels,
            event_index=720,
            observation_window_size=360,
            output_dir=cutoff_dir,
            run_stamp=run_stamp,
            inverse_transform_fn=boxcox_inverse,
            method_name=method_name,
            features_to_vary=features_to_vary,
            features_to_vary_label=cutoff,
            num_per_class=50,
            num_cfs_per_instance=10,
            random_state=42,
        )
        counterfactual_summary_path.parent.mkdir(parents=True, exist_ok=True)
        counterfactual_df.to_csv(counterfactual_summary_path, index=False)
        counterfactual_metrics_path = (
            output_root
            / method_name
            / cutoff
            / f"experiment19_{method_name}_{cutoff}_counterfactual_metrics_{run_stamp}.csv"
        )
        counterfactual_metrics_df = summarize_counterfactual_quality(counterfactual_df)
        counterfactual_metrics_df.to_csv(counterfactual_metrics_path, index=False)
        gallery_path = output_root / method_name / cutoff / f"experiment19_{method_name}_{cutoff}_counterfactual_gallery_{run_stamp}.png"
        plot_counterfactual_gallery(method_name=f"{method_name}:{cutoff}", plot_records=plot_records, output_path=gallery_path)
        method_results.append(
            {
                "method": method_name,
                "counterfactual_cutoff": cutoff,
                **provenance,
                "counterfactual_summary_path": counterfactual_summary_path,
                "counterfactual_metrics_path": counterfactual_metrics_path,
                "counterfactual_gallery_path": gallery_path,
                "best_row": best_row.to_dict(),
            }
        )
    print("Experiment 19:box_cox variant completed.\n")

    ab_summary = pd.DataFrame(method_results)
    ab_summary_path = output_root / f"experiment19_method_compare_{run_stamp}.csv"
    ab_summary_path.parent.mkdir(parents=True, exist_ok=True)
    ab_summary.to_csv(ab_summary_path, index=False)

    print("\n" + "=" * 80)
    print("Experiment 19 finished.")
    print(f"Method comparison summary: {ab_summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
