"""
Experiment 17: A/B baseline vs exp-before-FFT pipeline.

Two variants are trained and compared on the same splits:
- baseline: raw observation window -> FFT features -> RandomForest -> counterfactuals (same as experiment16)
- exp_fft: exp(raw window) -> FFT features -> RandomForest -> counterfactuals; plots use log scale to visualize reconstructions

Outputs are written under data/reports/experiment17/<variant>/.
The code intentionally reuses experiment16 utilities to keep behaviour aligned with the prior run.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import numpy as np

from experiment15 import TimeSeriesDataset, percent_key
from experiment16 import (
    generate_counterfactual_reports,
    get_splits_with_ids,
    plot_features_needed_for_half_importance,
    plot_toppercent_performance_by_lag,
    plot_toppercent_performance_by_window,
    run_independent_window_top_percent_experiment,
    save_model_artifacts,
    select_best,
    select_best_per_lag,
    select_best_window_per_lag_toppercent,
)


# Suppress all warnings to match experiment16 behaviour
warnings.filterwarnings("ignore")


_CHANNEL_MAX = {
    "p3_flux_ic": 43500.0,
    "p5_flux_ic": 2780.0,
    "p7_flux_ic": 652.0,
    "long": 46714.3,
}
_TARGET_PREEXP_MAX = 10.0  # keeps exp well below float32 overflow
_CHANNEL_SCALE = {k: (_TARGET_PREEXP_MAX / v) for k, v in _CHANNEL_MAX.items()}


def safe_exp_transform(sample: np.ndarray, channel_order: list[str]) -> np.ndarray:
    """Scale per-channel then exponentiate to avoid inf/overflow."""
    arr = np.asarray(sample, dtype=np.float64).copy()
    for idx, ch in enumerate(channel_order):
        scale = _CHANNEL_SCALE.get(ch, 1.0)
        arr[:, idx] *= scale
    # Optional extra guard; values above ~20 lead to huge exp but still finite
    arr = np.clip(arr, None, 20.0)
    return np.exp(arr, dtype=np.float64).astype(np.float32)


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

    results_path = out_dir / f"experiment17_{variant_name}_results_{variant_stamp}.csv"
    selection_path = out_dir / f"experiment17_{variant_name}_selection_{variant_stamp}.csv"
    results_df.to_csv(results_path, index=False)
    selection_df.to_csv(selection_path, index=False)

    models_dir = out_dir / "saved_models"
    saved_model_paths = save_model_artifacts_v17(
        model_artifacts=model_artifacts,
        output_dir=models_dir,
        run_stamp=variant_stamp,
        prefix=f"experiment17_{variant_name}",
    )

    val_plot_by_lag_path = out_dir / f"experiment17_{variant_name}_val_css_by_lag_{variant_stamp}.png"
    plot_toppercent_performance_by_lag(results_df, val_plot_by_lag_path, metric_col="val_css")

    test_plot_by_lag_path = out_dir / f"experiment17_{variant_name}_test_css_by_lag_{variant_stamp}.png"
    plot_toppercent_performance_by_lag(results_df, test_plot_by_lag_path, metric_col="test_css")

    val_plot_by_window_path = out_dir / f"experiment17_{variant_name}_val_css_by_window_{variant_stamp}.png"
    plot_toppercent_performance_by_window(results_df, val_plot_by_window_path, metric_col="val_css")

    test_plot_by_window_path = out_dir / f"experiment17_{variant_name}_test_css_by_window_{variant_stamp}.png"
    plot_toppercent_performance_by_window(results_df, test_plot_by_window_path, metric_col="test_css")

    feature_count_plot_path = out_dir / f"experiment17_{variant_name}_features_required_cumimp_gt50_{variant_stamp}.png"
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
    best_summary_path = out_dir / f"experiment17_{variant_name}_best_model_summary_{variant_stamp}.csv"
    pd.DataFrame([best_row]).to_csv(best_summary_path, index=False)

    best_rows_by_lag = select_best_per_lag(results_df)
    best_artifacts_by_lag = []
    for _, row in best_rows_by_lag.iterrows():
        artifact = next(
            candidate
            for candidate in model_artifacts
            if int(candidate["forecast_lag_min"]) == int(row["forecast_lag_min"])
            and int(candidate["window_size"]) == int(row["window_size"])
            and int(candidate["top_percent_key"]) == int(percent_key(float(row["top_percent"])))
        )
        best_artifacts_by_lag.append(artifact)

    counterfactual_plot_dir = out_dir / "counterfactual_timeseries" / variant_stamp
    counterfactual_summary_path = out_dir / f"experiment17_{variant_name}_counterfactual_summary_{variant_stamp}.csv"
    counterfactual_frames: list[pd.DataFrame] = []
    for artifact in best_artifacts_by_lag:
        counterfactual_frames.append(
            generate_counterfactual_reports(
                artifact=artifact,
                X_train=X_train,
                y_train=y_train,
                X_examples=X_test,
                y_examples=y_test,
                example_ids=test_ids,
                classes=classes,
                channel_names=channel_names,
                event_index=event_onset_index,
                observation_window_size=observation_window_size,
                output_dir=counterfactual_plot_dir,
                run_stamp=variant_stamp,
                random_state=42,
                chosen_sample_ids=[],
            )
        )
    counterfactual_df = (
        pd.concat(counterfactual_frames, axis=0, ignore_index=True) if counterfactual_frames else pd.DataFrame()
    )
    counterfactual_df.to_csv(counterfactual_summary_path, index=False)

    return {
        "variant": variant_name,
        "results_path": results_path,
        "selection_path": selection_path,
        "best_summary_path": best_summary_path,
        "counterfactual_summary_path": counterfactual_summary_path,
        "models_dir": models_dir,
        "saved_model_paths": saved_model_paths,
        "results_df": results_df,
        "best_row": best_row,
    }


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

    # Variant data copies
    X_train_exp = [safe_exp_transform(x, target_channels) for x in X_train]
    X_val_exp = [safe_exp_transform(x, target_channels) for x in X_val]
    X_test_exp = [safe_exp_transform(x, target_channels) for x in X_test]

    event_onset_index = 720
    observation_window_size = 360
    fft_window_sizes = [45, 90, 180, 360]
    forecast_lags = [5, 15, 30, 60, 120]
    top_percents = [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]

    output_root = data_root / "reports" / "experiment17"
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Baseline pipeline (raw)
    baseline_artifacts = run_variant(
        variant_name="baseline",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_ids=test_ids,
        classes=dataset.classes_,
        channel_names=target_channels,
        event_onset_index=event_onset_index,
        observation_window_size=observation_window_size,
        fft_window_sizes=fft_window_sizes,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        output_root=output_root,
        run_stamp=run_stamp,
    )

    # Exp-before-FFT pipeline
    exp_fft_artifacts = run_variant(
        variant_name="exp_fft",
        X_train=X_train_exp,
        y_train=y_train,
        X_val=X_val_exp,
        y_val=y_val,
        X_test=X_test_exp,
        y_test=y_test,
        test_ids=test_ids,
        classes=dataset.classes_,
        channel_names=target_channels,
        event_onset_index=event_onset_index,
        observation_window_size=observation_window_size,
        fft_window_sizes=fft_window_sizes,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        output_root=output_root,
        run_stamp=run_stamp,
    )

    # A/B summary
    ab_summary = pd.DataFrame(
        [
            {"variant": baseline_artifacts["variant"], **baseline_artifacts["best_row"].to_dict()},
            {"variant": exp_fft_artifacts["variant"], **exp_fft_artifacts["best_row"].to_dict()},
        ]
    )
    ab_summary_path = output_root / f"experiment17_ab_compare_{run_stamp}.csv"
    ab_summary.to_csv(ab_summary_path, index=False)

    print("\n" + "=" * 80)
    print("Experiment 17 finished.")
    print(f"Baseline best summary: {baseline_artifacts['best_summary_path']}")
    print(f"Exp-FFT best summary: {exp_fft_artifacts['best_summary_path']}")
    print(f"A/B summary: {ab_summary_path}")
    print(f"Baseline models dir: {baseline_artifacts['models_dir']}")
    print(f"Exp-FFT models dir: {exp_fft_artifacts['models_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
