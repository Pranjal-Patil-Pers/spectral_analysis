"""
Experiment 17: compare positivity-preserving transforms for counterfactual generation.

Each method applies a forward transform to the input window, builds FFT features,
trains a RandomForest, generates counterfactuals, reconstructs through IFFT, then
applies the inverse transform before plotting the time series.
"""

from __future__ import annotations

import pickle
import warnings
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import numpy as np

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


def safe_exp_transform(sample: np.ndarray, channel_order: list[str]) -> np.ndarray:
    """Scale per-channel then exponentiate to avoid inf/overflow."""
    arr = np.asarray(sample, dtype=np.float64).copy()
    for idx, ch in enumerate(channel_order):
        scale = _CHANNEL_SCALE.get(ch, 1.0)
        arr[:, idx] *= scale
    arr = np.clip(arr, None, 20.0)
    return np.exp(arr, dtype=np.float64).astype(np.float32)


def softplus_transform(sample: np.ndarray) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float64)
    return np.log1p(np.exp(np.clip(arr, None, 20.0))).astype(np.float32)


def square_transform(sample: np.ndarray) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float64)
    return np.square(arr).astype(np.float32)


def abs_transform(sample: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(sample, dtype=np.float64)).astype(np.float32)


def inverse_exp(sample: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(sample, dtype=np.float64), 1e-12, None)).astype(np.float32)


def inverse_softplus(sample: np.ndarray) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float64)
    return np.log(np.expm1(np.clip(arr, 1e-12, None))).astype(np.float32)


def inverse_square(sample: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(np.asarray(sample, dtype=np.float64), 0.0, None)).astype(np.float32)


def inverse_abs(sample: np.ndarray) -> np.ndarray:
    return _as_float32(sample)


def fit_yeo_johnson_transform(X_train: list[np.ndarray]) -> tuple[PowerTransformer, callable, callable]:
    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    flat = np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1, 1) for x in X_train], axis=0)
    transformer.fit(flat)

    def forward(sample: np.ndarray) -> np.ndarray:
        shape = np.asarray(sample).shape
        transformed = transformer.transform(np.asarray(sample, dtype=np.float64).reshape(-1, 1))
        return transformed.reshape(shape).astype(np.float32)

    def inverse(sample: np.ndarray) -> np.ndarray:
        shape = np.asarray(sample).shape
        restored = transformer.inverse_transform(np.asarray(sample, dtype=np.float64).reshape(-1, 1))
        return restored.reshape(shape).astype(np.float32)

    return transformer, forward, inverse


def transform_dataset(X_data: list[np.ndarray], transform_fn) -> list[np.ndarray]:
    return [transform_fn(x) for x in X_data]


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
    out_dir.mkdir(parents=True, exist_ok=True)
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
    num_per_class: int = 3,
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

    rows: list[dict[str, float | int | str]] = []
    plot_records: list[dict[str, object]] = []
    channels = int(train_feature_bank["channels"])
    n_slices = int(train_feature_bank["n_slices"])
    max_coeffs = int(train_feature_bank["max_coeffs"])
    fft_window_size = int(artifact["window_size"])
    lag = int(artifact["forecast_lag_min"])
    method_dir = output_dir / method_name

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
        target_label = 1 - int(label)
        target_label_name = str(classes[int(target_label)])

        try:
            cf_result = dice.generate_counterfactuals(
                query_df,
                total_CFs=1,
                desired_class="opposite",
                features_to_vary="all",
                verbose=False,
                sample_size=2000,
                random_seed=random_state + idx,
            )
            cf_df = cf_result.cf_examples_list[0].final_cfs_df
            if len(cf_df) == 0:
                raise RuntimeError("DiCE returned no counterfactual rows.")
            cf_row = cf_df.iloc[0]
            cf_selected = cf_row[feature_names].astype(float).to_numpy(dtype=np.float64)
            cf_prediction = int(model.predict(cf_selected.reshape(1, -1))[0])
            cf_full = original_full.copy()
            cf_full[selected_idx] = cf_selected

            original_obs = inverse_transform_fn(
                extract_observation_window(
                    X_examples[idx],
                    event_index=event_index,
                    observation_window_size=observation_window_size,
                    lag_minute=lag,
                )
            )
            original_recon = inverse_transform_fn(
                reconstruct_observation_from_fft_features(
                    original_full,
                    channels=channels,
                    n_slices=n_slices,
                    max_coeffs=max_coeffs,
                    fft_window_size=fft_window_size,
                )
            )
            cf_recon = inverse_transform_fn(
                reconstruct_observation_from_fft_features(
                    cf_full,
                    channels=channels,
                    n_slices=n_slices,
                    max_coeffs=max_coeffs,
                    fft_window_size=fft_window_size,
                )
            )
            original_mse = float(np.mean((original_obs - original_recon) ** 2))
            counterfactual_mse = float(np.mean((original_obs - cf_recon) ** 2))
            plot_records.append(
                {
                    "sample_id": sample_id,
                    "label_name": label_name,
                    "target_label_name": target_label_name,
                    "original_obs": original_obs,
                    "original_recon": original_recon,
                    "cf_recon": cf_recon,
                    "lag": lag,
                    "window_size": fft_window_size,
                    "top_percent_key": int(artifact["top_percent_key"]),
                    "original_mse": original_mse,
                    "counterfactual_mse": counterfactual_mse,
                    "channel_names": channel_names,
                }
            )

            combined_plot_path = method_dir / "combined" / (
                f"experiment17_{method_name}_cf_sample_{sample_id}_lag{lag}_w{fft_window_size}_"
                f"top{int(artifact['top_percent_key'])}_{label_name}_to_{target_label_name}_{run_stamp}_combined.png"
            )
            plot_counterfactual_timeseries(
                sample_id=sample_id,
                label_name=f"{method_name}: {label_name}",
                target_label_name=target_label_name,
                channel_names=channel_names,
                original_obs=original_obs,
                original_recon=original_recon,
                cf_recon=cf_recon,
                lag=lag,
                window_size=fft_window_size,
                top_percent_key=int(artifact["top_percent_key"]),
                original_mse=original_mse,
                counterfactual_mse=counterfactual_mse,
                output_path=combined_plot_path,
            )
            for c in range(channels):
                channel_label = channel_names[c] if c < len(channel_names) else f"ch{c}"
                channel_plot_path = method_dir / "channels" / (
                    f"experiment17_{method_name}_cf_sample_{sample_id}_lag{lag}_w{fft_window_size}_"
                    f"top{int(artifact['top_percent_key'])}_{channel_label}_{label_name}_"
                    f"to_{target_label_name}_{run_stamp}.png"
                )
                plot_counterfactual_channel_timeseries(
                    sample_id=sample_id,
                    label_name=f"{method_name}: {label_name}",
                    target_label_name=target_label_name,
                    channel_name=channel_label,
                    original_obs_channel=original_obs[:, c],
                    cf_recon_channel=cf_recon[:, c],
                    lag=lag,
                    window_size=fft_window_size,
                    top_percent_key=int(artifact["top_percent_key"]),
                    counterfactual_mse=float(np.mean((original_obs[:, c] - cf_recon[:, c]) ** 2)),
                    output_path=channel_plot_path,
                )
            cf_generated = True
        except Exception as exc:
            error_message = str(exc)

        rows.append(
            {
                "method": method_name,
                "sample_id": sample_id,
                "forecast_lag_min": int(lag),
                "window_size": int(fft_window_size),
                "top_percent": float(artifact["top_percent"]),
                "top_percent_label": f"{int(artifact['top_percent_key'])}%",
                "original_label": int(label),
                "original_label_name": label_name,
                "desired_counterfactual_label": int(target_label),
                "desired_counterfactual_label_name": target_label_name,
                "counterfactual_predicted_label": int(cf_prediction),
                "counterfactual_predicted_label_name": str(classes[int(cf_prediction)]),
                "counterfactual_found": bool(cf_generated),
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

        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{record['sample_id']} | {record['label_name']} -> {record['target_label_name']}\n"
            f"MSE={float(record['counterfactual_mse']):.3e}",
            fontsize=10,
        )
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    for j in range(n_plots, len(axes_arr)):
        axes_arr[j].axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"Experiment 17 Counterfactual Gallery | {method_name}", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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

    method_specs = {
        "exp": {
            "forward": lambda x: safe_exp_transform(x, target_channels),
            "inverse": inverse_exp,
        },
        "softplus": {
            "forward": softplus_transform,
            "inverse": inverse_softplus,
        },
        "square": {
            "forward": square_transform,
            "inverse": inverse_square,
        },
        "abs": {
            "forward": abs_transform,
            "inverse": inverse_abs,
        },
        "yeo_johnson": {},
    }
    _yj_transformer, yj_forward, yj_inverse = fit_yeo_johnson_transform(X_train)
    method_specs["yeo_johnson"]["forward"] = yj_forward
    method_specs["yeo_johnson"]["inverse"] = yj_inverse

    event_onset_index = 720
    observation_window_size = 360
    fft_window_sizes = [45, 90, 180, 360]
    forecast_lags = [5, 15, 30, 60, 120]
    top_percents = [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]

    output_root = data_root / "reports" / "experiment17"
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    method_results: list[dict[str, object]] = []
    for method_name, spec in method_specs.items():
        print(f"Experiment 17:{method_name} variant starting...")
        X_train_m = transform_dataset(X_train, spec["forward"])
        X_val_m = transform_dataset(X_val, spec["forward"])
        X_test_m = transform_dataset(X_test, spec["forward"])

        method_artifacts = run_variant(
            variant_name=method_name,
            X_train=X_train_m,
            y_train=y_train,
            X_val=X_val_m,
            y_val=y_val,
            X_test=X_test_m,
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

        best_row = select_best(method_artifacts["results_df"])
        best_artifact = None
        for path in method_artifacts["saved_model_paths"]:
            with open(path, "rb") as f:
                artifact = pickle.load(f)
            if (
                int(artifact["forecast_lag_min"]) == int(best_row["forecast_lag_min"])
                and int(artifact["window_size"]) == int(best_row["window_size"])
                and int(artifact["top_percent_key"]) == int(percent_key(float(best_row["top_percent"])))
            ):
                best_artifact = artifact
                break
        if best_artifact is None:
            raise RuntimeError(f"Could not find best artifact for method '{method_name}'.")
        counterfactual_plot_dir = output_root / method_name / "counterfactual_timeseries" / run_stamp
        counterfactual_summary_path = output_root / method_name / f"experiment17_{method_name}_counterfactual_summary_{run_stamp}.csv"
        counterfactual_df, plot_records = generate_transform_counterfactual_reports(
            artifact=best_artifact,
            X_train=X_train_m,
            y_train=y_train,
            X_examples=X_test_m,
            y_examples=y_test,
            example_ids=test_ids,
            classes=dataset.classes_,
            channel_names=target_channels,
            event_index=event_onset_index,
            observation_window_size=observation_window_size,
            output_dir=counterfactual_plot_dir,
            run_stamp=run_stamp,
            inverse_transform_fn=spec["inverse"],
            method_name=method_name,
            num_per_class=3,
            random_state=42,
        )
        counterfactual_summary_path.parent.mkdir(parents=True, exist_ok=True)
        counterfactual_df.to_csv(counterfactual_summary_path, index=False)
        gallery_path = output_root / method_name / f"experiment17_{method_name}_counterfactual_gallery_{run_stamp}.png"
        plot_counterfactual_gallery(method_name=method_name, plot_records=plot_records, output_path=gallery_path)
        method_results.append(
            {
                "method": method_name,
                "results_path": method_artifacts["results_path"],
                "selection_path": method_artifacts["selection_path"],
                "best_summary_path": method_artifacts["best_summary_path"],
                "counterfactual_summary_path": counterfactual_summary_path,
                "counterfactual_gallery_path": gallery_path,
                "best_row": best_row.to_dict(),
            }
        )
        print(f"Experiment 17:{method_name} variant completed.\n")

    ab_summary = pd.DataFrame(method_results)
    ab_summary_path = output_root / f"experiment17_method_compare_{run_stamp}.csv"
    ab_summary_path.parent.mkdir(parents=True, exist_ok=True)
    ab_summary.to_csv(ab_summary_path, index=False)

    print("\n" + "=" * 80)
    print("Experiment 17 finished.")
    print(f"Method comparison summary: {ab_summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
