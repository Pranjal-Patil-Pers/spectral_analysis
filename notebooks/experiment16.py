"""
Experiment 16:
- Keep FFT windows independent; do not concatenate windows.
- For each (lag, window), train a baseline RandomForest on all FFT features.
- Select top-k% features inside that single window-specific model.
- Train one RandomForest per (lag, window, top%).
- Plot all lag/window/top% results and select the best model by validation priority.
- Generate DiCE counterfactuals for 2 training examples per class on the best model.
- Reconstruct original and counterfactual FFT features back to time series via inverse FFT.
- Plot original vs. reconstructed-via-counterfactual time series and report reconstruction errors.
"""

from __future__ import annotations
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
import math
import pickle
from pathlib import Path

import dice_ml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import irfft
from sklearn.ensemble import RandomForestClassifier

from experiment15 import (
    TimeSeriesDataset,
    build_fft_feature_names,
    build_fft_features_all,
    count_features_for_cumulative_importance,
    compute_all_metrics,
    extract_observation_window,
    percent_key,
    pick_positive_class,
    plot_features_needed_for_half_importance,
    select_top_percentage_indices,
)


def get_splits_with_ids(
    dataset: TimeSeriesDataset,
) -> tuple[
    tuple[list[np.ndarray], np.ndarray, list[str]],
    tuple[list[np.ndarray], np.ndarray, list[str]],
    tuple[list[np.ndarray], np.ndarray, list[str]],
]:
    train_X, train_y, train_ids = [], [], []
    val_X, val_y, val_ids = [], [], []
    test_X, test_y, test_ids = [], [], []

    for idx in range(len(dataset)):
        row = dataset.metadata.iloc[idx]
        year = int(row["event_time"].year)

        try:
            series, label = dataset[idx]
        except Exception as exc:
            print(f"Skipping sample index {idx}: {exc}")
            continue

        sample_id = str(Path(str(row[dataset.filename_col])).stem)
        if year <= 1992:
            train_X.append(series)
            train_y.append(label)
            train_ids.append(sample_id)
        elif 1992 < year <= 2002:
            val_X.append(series)
            val_y.append(label)
            val_ids.append(sample_id)
        elif 2002 < year <= 2018:
            test_X.append(series)
            test_y.append(label)
            test_ids.append(sample_id)

    if len(train_X) == 0 or len(val_X) == 0 or len(test_X) == 0:
        raise ValueError(
            "One or more splits are empty. "
            f"train={len(train_X)}, val={len(val_X)}, test={len(test_X)}."
        )

    y_train = np.asarray(train_y, dtype=np.int32)
    y_val = np.asarray(val_y, dtype=np.int32)
    y_test = np.asarray(test_y, dtype=np.int32)

    print(
        "Year-based split counts: "
        f"train={len(train_X)} (<=1992), "
        f"val={len(val_X)} (1992,2002], "
        f"test={len(test_X)} (2002,2018]"
    )

    return (
        (train_X, y_train, train_ids),
        (val_X, y_val, val_ids),
        (test_X, y_test, test_ids),
    )


def sort_by_validation_priority(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "val_css",
            "val_tss",
            "val_hss",
            "val_f1",
            "selected_feature_count",
            "test_css",
            "test_tss",
            "test_hss",
            "test_f1",
        ],
        ascending=[False, False, False, False, True, False, False, False, False],
    )


def select_best(results_df: pd.DataFrame) -> pd.Series:
    return sort_by_validation_priority(results_df).iloc[0]


def select_best_per_lag(results_df: pd.DataFrame) -> pd.DataFrame:
    if len(results_df) == 0:
        return results_df.copy()

    best_rows: list[pd.Series] = []
    grouped = results_df.groupby("forecast_lag_min", sort=True, dropna=False)
    for _, group_df in grouped:
        best_rows.append(select_best(group_df))
    return pd.DataFrame(best_rows).reset_index(drop=True)


def extract_run_stamp(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unable to extract run stamp from '{path.name}'.")
    return "_".join(parts[-2:])


def find_best_cached_run(output_dir: Path) -> dict | None:
    best_cached: dict | None = None
    results_paths = sorted(output_dir.glob("experiment16_window_independent_toppercent_results_*.csv"))

    for results_path in results_paths:
        try:
            run_stamp = extract_run_stamp(results_path)
            results_df = pd.read_csv(results_path)
            if len(results_df) == 0:
                continue
            best_row = select_best(results_df)

            selection_path = output_dir / f"experiment16_window_independent_selection_details_{run_stamp}.csv"
            summary_path = output_dir / f"experiment16_best_model_summary_{run_stamp}.csv"
            model_path = (
                output_dir
                / "saved_models"
                / (
                    f"experiment16_rf_w{int(best_row['window_size'])}_"
                    f"lag{int(best_row['forecast_lag_min'])}_"
                    f"top{int(percent_key(float(best_row['top_percent'])))}_{run_stamp}.pkl"
                )
            )

            if not (results_path.exists() and selection_path.exists() and model_path.exists()):
                continue

            with open(model_path, "rb") as f:
                best_artifact = pickle.load(f)
            model = best_artifact.get("model")
            if model is not None and hasattr(model, "set_params"):
                try:
                    model.set_params(n_jobs=1)
                except Exception:
                    pass

            candidate = {
                "run_stamp": run_stamp,
                "summary_path": summary_path,
                "results_path": results_path,
                "selection_path": selection_path,
                "model_path": model_path,
                "best_row": best_row,
                "best_artifact": best_artifact,
                "results_df": results_df,
                "selection_df": pd.read_csv(selection_path),
            }
            if best_cached is None:
                best_cached = candidate
            else:
                compare_df = pd.DataFrame([best_cached["best_row"], candidate["best_row"]]).copy()
                compare_df["__candidate_id"] = [0, 1]
                ranked = sort_by_validation_priority(compare_df).reset_index(drop=True)
                if int(ranked.iloc[0]["__candidate_id"]) == 1:
                    best_cached = candidate
        except Exception:
            continue

    return best_cached


def load_artifact_for_row(output_dir: Path, run_stamp: str, row: pd.Series) -> dict:
    model_path = (
        output_dir
        / "saved_models"
        / (
            f"experiment16_rf_w{int(row['window_size'])}_"
            f"lag{int(row['forecast_lag_min'])}_"
            f"top{int(percent_key(float(row['top_percent'])))}_{run_stamp}.pkl"
        )
    )
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact.get("model")
    if model is not None and hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass
    return artifact


def select_best_window_per_lag_toppercent(results_df: pd.DataFrame) -> pd.DataFrame:
    if len(results_df) == 0:
        return results_df.copy()

    grouped_rows: list[pd.Series] = []
    grouped = results_df.groupby(["forecast_lag_min", "top_percent"], sort=True, dropna=False)
    for _, group_df in grouped:
        grouped_rows.append(sort_by_validation_priority(group_df).iloc[0])

    if not grouped_rows:
        return results_df.iloc[0:0].copy()

    return pd.DataFrame(grouped_rows).reset_index(drop=True)


def run_independent_window_top_percent_experiment(
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_val: list[np.ndarray],
    y_val: np.ndarray,
    X_test: list[np.ndarray],
    y_test: np.ndarray,
    classes: np.ndarray,
    channel_names: list[str],
    fft_window_sizes: list[int],
    forecast_lags: list[int],
    top_percents: list[float],
    event_index: int,
    observation_window_size: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    positive_class = pick_positive_class(classes)
    print(f"Positive class for scoring: idx={positive_class}, label='{classes[positive_class]}'")

    results_rows: list[dict[str, float | int | str]] = []
    selection_rows: list[dict[str, float | int | str]] = []
    model_artifacts: list[dict] = []

    for lag in forecast_lags:
        print(f"\n{'=' * 80}")
        print(f"Lag: {lag}")

        for fft_window_size in fft_window_sizes:
            input_end = event_index - lag
            input_start = input_end - observation_window_size
            print(
                f"Preparing lag={lag}, window={fft_window_size} | input=[{input_start}:{input_end}]"
            )

            X_train_all, max_coeffs, n_slices, channels = build_fft_features_all(
                X_train,
                fft_window_size=fft_window_size,
                event_index=event_index,
                observation_window_size=observation_window_size,
                forecast_lag=lag,
            )
            X_val_all, _, _, _ = build_fft_features_all(
                X_val,
                fft_window_size=fft_window_size,
                event_index=event_index,
                observation_window_size=observation_window_size,
                forecast_lag=lag,
            )
            X_test_all, _, _, _ = build_fft_features_all(
                X_test,
                fft_window_size=fft_window_size,
                event_index=event_index,
                observation_window_size=observation_window_size,
                forecast_lag=lag,
            )

            X_train_flat = X_train_all.reshape(X_train_all.shape[0], -1)
            X_val_flat = X_val_all.reshape(X_val_all.shape[0], -1)
            X_test_flat = X_test_all.reshape(X_test_all.shape[0], -1)

            feature_names = build_fft_feature_names(
                window_size=fft_window_size,
                lag=lag,
                channels=channels,
                channel_names=channel_names,
                n_slices=n_slices,
                max_coeffs=max_coeffs,
            )
            if len(feature_names) != int(X_train_flat.shape[1]):
                raise ValueError(
                    f"Feature name count mismatch for window={fft_window_size}, lag={lag}: "
                    f"{len(feature_names)} vs {X_train_flat.shape[1]}"
                )

            baseline_model = RandomForestClassifier(
                n_estimators=300,
                random_state=random_state + int(lag) * 1000 + int(fft_window_size),
                n_jobs=1,
            )
            baseline_model.fit(X_train_flat, y_train)
            baseline_importances = np.asarray(baseline_model.feature_importances_, dtype=np.float64)
            total_features = int(baseline_importances.size)

            for frac in top_percents:
                pkey = percent_key(frac)
                selected_idx, selected_count = select_top_percentage_indices(baseline_importances, frac)

                X_train_sel = X_train_flat[:, selected_idx]
                X_val_sel = X_val_flat[:, selected_idx]
                X_test_sel = X_test_flat[:, selected_idx]
                selected_feature_names = [feature_names[int(i)] for i in selected_idx.tolist()]

                model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state + int(lag) * 1000 + int(fft_window_size) * 10 + int(pkey),
                    n_jobs=1,
                )
                model.fit(X_train_sel, y_train)

                y_pred_train = model.predict(X_train_sel)
                y_pred_val = model.predict(X_val_sel)
                y_pred_test = model.predict(X_test_sel)

                train_metrics = compute_all_metrics(y_train, y_pred_train, positive_class)
                val_metrics = compute_all_metrics(y_val, y_pred_val, positive_class)
                test_metrics = compute_all_metrics(y_test, y_pred_test, positive_class)
                model_importances = np.asarray(model.feature_importances_, dtype=np.float64)
                model_half_count, model_half_reached = count_features_for_cumulative_importance(
                    model_importances,
                    threshold=0.50,
                    strict=True,
                )

                results_rows.append(
                    {
                        "forecast_lag_min": int(lag),
                        "window_size": int(fft_window_size),
                        "top_percent": float(frac),
                        "top_percent_label": f"{pkey}%",
                        "input_start": int(input_start),
                        "input_end": int(input_end),
                        "total_features": int(total_features),
                        "selected_feature_count": int(selected_count),
                        "selected_feature_fraction": float(selected_count) / float(total_features),
                        "feature_count_cumimp_gt50": int(model_half_count),
                        "cumimp_reached_at_gt50": float(model_half_reached),
                        "train_accuracy": train_metrics["accuracy"],
                        "train_f1": train_metrics["f1"],
                        "train_precision": train_metrics["precision"],
                        "train_recall": train_metrics["recall"],
                        "train_tss": train_metrics["tss"],
                        "train_hss": train_metrics["hss"],
                        "train_css": train_metrics["css"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_f1": val_metrics["f1"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_tss": val_metrics["tss"],
                        "val_hss": val_metrics["hss"],
                        "val_css": val_metrics["css"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_f1": test_metrics["f1"],
                        "test_precision": test_metrics["precision"],
                        "test_recall": test_metrics["recall"],
                        "test_tss": test_metrics["tss"],
                        "test_hss": test_metrics["hss"],
                        "test_css": test_metrics["css"],
                    }
                )

                selection_rows.append(
                    {
                        "forecast_lag_min": int(lag),
                        "window_size": int(fft_window_size),
                        "top_percent": float(frac),
                        "top_percent_label": f"{pkey}%",
                        "selection_scope": "within_single_window_model",
                        "total_features_in_window_model": int(total_features),
                        "selected_feature_count": int(selected_count),
                        "selected_feature_fraction": float(selected_count) / float(total_features),
                        "max_available_coeffs": int(max_coeffs),
                        "n_slices": int(n_slices),
                        "channels": int(channels),
                    }
                )

                model_artifacts.append(
                    {
                        "forecast_lag_min": int(lag),
                        "window_size": int(fft_window_size),
                        "top_percent": float(frac),
                        "top_percent_key": int(pkey),
                        "model": model,
                        "selected_feature_indices": [int(i) for i in selected_idx.tolist()],
                        "selected_feature_names": selected_feature_names,
                        "all_feature_names": feature_names,
                        "max_coeffs": int(max_coeffs),
                        "n_slices": int(n_slices),
                        "channels": int(channels),
                        "total_features": int(total_features),
                    }
                )

                print(
                    f"Trained model lag{lag}_w{fft_window_size}_top{pkey}% | "
                    f"selected={selected_count} | val_css={val_metrics['css']:.4f} | "
                    f"test_css={test_metrics['css']:.4f}"
                )

    if not results_rows:
        raise RuntimeError("No model results generated.")

    return pd.DataFrame(results_rows), pd.DataFrame(selection_rows), model_artifacts


def plot_toppercent_performance_by_lag(
    results_df: pd.DataFrame,
    output_path: Path,
    metric_col: str,
):
    if len(results_df) == 0:
        return

    lags = sorted(results_df["forecast_lag_min"].astype(int).unique().tolist())
    if not lags:
        return

    n_cols = min(3, len(lags))
    n_rows = int(math.ceil(float(len(lags)) / float(n_cols)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.2 * n_cols, 4.8 * n_rows),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.flatten()

    for lag_idx, lag in enumerate(lags):
        ax = flat_axes[lag_idx]
        lag_df = results_df[results_df["forecast_lag_min"].astype(int) == int(lag)].copy()
        windows = sorted(lag_df["window_size"].astype(int).unique().tolist())
        top_percents = sorted(lag_df["top_percent"].astype(float).unique().tolist())
        colors = plt.get_cmap("tab10", max(1, len(windows)))
        x = np.arange(len(top_percents), dtype=np.float64)
        x_labels = [f"{int(round(p * 100.0))}%" for p in top_percents]

        for color_idx, window in enumerate(windows):
            curve_df = (
                lag_df[lag_df["window_size"].astype(int) == int(window)]
                .sort_values("top_percent")
            )
            ax.plot(
                x,
                curve_df[metric_col].astype(float).values,
                marker="o",
                linewidth=1.8,
                color=colors(color_idx),
                label=f"W{int(window)}",
            )

        ax.set_title(f"Lag {int(lag)} min")
        ax.set_xlabel("Top-% selected within window")
        ax.set_ylabel(metric_col.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        if lag_idx == 0:
            ax.legend(frameon=False, fontsize=9, loc="best")

    for ax in flat_axes[len(lags):]:
        ax.axis("off")

    fig.suptitle(f"Experiment 16: {metric_col.upper()} by lag and window", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_toppercent_performance_by_window(
    results_df: pd.DataFrame,
    output_path: Path,
    metric_col: str,
):
    if len(results_df) == 0:
        return

    windows = sorted(results_df["window_size"].astype(int).unique().tolist())
    if not windows:
        return

    n_cols = min(2, len(windows))
    n_rows = int(math.ceil(float(len(windows)) / float(n_cols)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.2 * n_cols, 4.8 * n_rows),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.flatten()

    for win_idx, window in enumerate(windows):
        ax = flat_axes[win_idx]
        win_df = results_df[results_df["window_size"].astype(int) == int(window)].copy()
        lags = sorted(win_df["forecast_lag_min"].astype(int).unique().tolist())
        top_percents = sorted(win_df["top_percent"].astype(float).unique().tolist())
        colors = plt.get_cmap("tab10", max(1, len(lags)))
        x = np.arange(len(top_percents), dtype=np.float64)
        x_labels = [f"{int(round(p * 100.0))}%" for p in top_percents]

        for color_idx, lag in enumerate(lags):
            curve_df = (
                win_df[win_df["forecast_lag_min"].astype(int) == int(lag)]
                .sort_values("top_percent")
            )
            ax.plot(
                x,
                curve_df[metric_col].astype(float).values,
                marker="o",
                linewidth=1.8,
                color=colors(color_idx),
                label=f"lag {int(lag)}",
            )

        ax.set_title(f"Window {int(window)} min")
        ax.set_xlabel("Top-% selected within window")
        ax.set_ylabel(metric_col.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        if win_idx == 0:
            ax.legend(frameon=False, fontsize=9, loc="best")

    for ax in flat_axes[len(windows):]:
        ax.axis("off")

    fig.suptitle(f"Experiment 16: {metric_col.upper()} by window and lag", fontsize=12, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_model_artifacts(model_artifacts: list[dict], output_dir: Path, run_stamp: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for artifact in model_artifacts:
        lag = int(artifact["forecast_lag_min"])
        window = int(artifact["window_size"])
        top_key = int(artifact["top_percent_key"])
        path = output_dir / f"experiment16_rf_w{window}_lag{lag}_top{top_key}_{run_stamp}.pkl"
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        saved_paths.append(path)

    return saved_paths


def reconstruct_observation_from_fft_features(
    flat_features: np.ndarray,
    channels: int,
    n_slices: int,
    max_coeffs: int,
    fft_window_size: int,
) -> np.ndarray:
    arr = np.asarray(flat_features, dtype=np.float64).reshape(2 * int(channels), int(n_slices) * int(max_coeffs))
    reconstructed = np.zeros((int(n_slices) * int(fft_window_size), int(channels)), dtype=np.float64)

    for s in range(int(n_slices)):
        a = int(s) * int(fft_window_size)
        b = a + int(fft_window_size)
        coeff_start = int(s) * int(max_coeffs)
        coeff_end = coeff_start + int(max_coeffs)

        for c in range(int(channels)):
            mag = arr[c, coeff_start:coeff_end].copy()
            phase = arr[int(channels) + c, coeff_start:coeff_end].copy()
            mag = np.maximum(mag, 0.0)
            phase = np.angle(np.exp(1j * phase))
            fft_vals = mag * np.exp(1j * phase)
            reconstructed[a:b, c] = irfft(fft_vals, n=int(fft_window_size))

    return reconstructed.astype(np.float32)


def build_best_model_feature_bank(
    artifact: dict,
    X_train: list[np.ndarray],
    event_index: int,
    observation_window_size: int,
) -> dict[str, np.ndarray | int]:
    lag = int(artifact["forecast_lag_min"])
    fft_window_size = int(artifact["window_size"])

    X_train_all, max_coeffs, n_slices, channels = build_fft_features_all(
        X_train,
        fft_window_size=fft_window_size,
        event_index=event_index,
        observation_window_size=observation_window_size,
        forecast_lag=lag,
    )

    return {
        "X_train_flat": X_train_all.reshape(X_train_all.shape[0], -1),
        "max_coeffs": int(max_coeffs),
        "n_slices": int(n_slices),
        "channels": int(channels),
    }


def choose_training_examples_for_counterfactuals(
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    train_ids: list[str],
    model: RandomForestClassifier,
    classes: np.ndarray,
    num_per_class: int = 2,
    chosen_sample_ids: list[str] | None = None,
) -> list[dict[str, int | str]]:
    preds = model.predict(X_train_sel)
    chosen: list[dict[str, int | str]] = []
    requested_ids = [str(sample_id) for sample_id in (chosen_sample_ids or []) if str(sample_id).strip()]

    if requested_ids:
        train_id_to_index = {str(sample_id): idx for idx, sample_id in enumerate(train_ids)}
        missing_ids = [sample_id for sample_id in requested_ids if sample_id not in train_id_to_index]
        if missing_ids:
            raise ValueError(
                "Chosen counterfactual example file(s) were not found in the training split: "
                + ", ".join(missing_ids)
            )

        for sample_id in requested_ids:
            idx = int(train_id_to_index[sample_id])
            chosen.append(
                {
                    "train_index": int(idx),
                    "sample_id": str(train_ids[int(idx)]),
                    "label": int(y_train[int(idx)]),
                    "label_name": str(classes[int(y_train[int(idx)])]),
                    "predicted_label": int(preds[int(idx)]),
                    "predicted_label_name": str(classes[int(preds[int(idx)])]),
                }
            )
        return chosen

    for cls_idx in sorted(np.unique(y_train).tolist()):
        cls_mask = y_train == int(cls_idx)
        correct_idx = np.where(cls_mask & (preds == y_train))[0].tolist()
        fallback_idx = np.where(cls_mask)[0].tolist()
        source = correct_idx if len(correct_idx) >= int(num_per_class) else fallback_idx

        for idx in source[: int(num_per_class)]:
            chosen.append(
                {
                    "train_index": int(idx),
                    "sample_id": str(train_ids[int(idx)]),
                    "label": int(y_train[int(idx)]),
                    "label_name": str(classes[int(y_train[int(idx)])]),
                    "predicted_label": int(preds[int(idx)]),
                    "predicted_label_name": str(classes[int(preds[int(idx)])]),
                }
            )

    return chosen


def make_dice_explainer(
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    model: RandomForestClassifier,
) -> dice_ml.Dice:
    train_df = pd.DataFrame(X_train_sel, columns=feature_names)
    train_df["target"] = y_train.astype(int)
    data_interface = dice_ml.Data(
        dataframe=train_df,
        continuous_features=feature_names,
        outcome_name="target",
    )
    model_interface = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier",
    )
    return dice_ml.Dice(data_interface, model_interface, method="random")



def plot_counterfactual_timeseries(
    sample_id: str,
    label_name: str,
    target_label_name: str,
    channel_names: list[str],
    original_obs: np.ndarray,
    original_recon: np.ndarray,
    cf_recon: np.ndarray,
    lag: int,
    window_size: int,
    top_percent_key: int,
    original_mse: float,
    counterfactual_mse: float,
    output_path: Path,
):
    channels = int(original_obs.shape[1])
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    x = np.arange(original_obs.shape[0], dtype=np.int64)
    custom_colors = ["#FF0000", "#1b5c0c", "#FFA500", "#0000FF"]
    colors = custom_colors

    for c in range(channels):
        label = channel_names[c] if c < len(channel_names) else f"ch{c}"
        color = colors[c % len(colors)]
        ax.plot(
            x,
            original_obs[:, c],
            color=color,
            linewidth=1.8,
            label=f"{label} original",
        )
        ax.plot(
            x,
            cf_recon[:, c],
            color=color,
            linewidth=1.8,
            linestyle="--",
            label=f"{label} counterfactual",
        )

    ax.set_xlabel("Minutes in observation window")
    ax.set_ylabel("Flux (pfu)")
    ax.set_yscale("log")
    positive_values = np.concatenate(
        [
            np.asarray(original_obs, dtype=np.float64).ravel(),
            np.asarray(cf_recon, dtype=np.float64).ravel(),
        ]
    )
    positive_values = positive_values[np.isfinite(positive_values) & (positive_values > 0)]
    if positive_values.size > 0:
        y_min = min(1e-2, float(np.min(positive_values)))
        y_max = float(np.max(positive_values))
        if y_max <= y_min:
            y_max = y_min * 10.0
        tick_candidates = np.array([0.1, 0.5, 1.0, y_max], dtype=np.float64)
        tick_candidates = tick_candidates[(tick_candidates >= y_min) & (tick_candidates <= y_max)]
        if tick_candidates.size == 0:
            tick_candidates = np.array([y_min, y_max], dtype=np.float64)
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.set_yticks(np.unique(np.round(tick_candidates, 6)))
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.suptitle(
        f"Counterfactual Time Series | sample={sample_id} | {label_name} -> {target_label_name} | "
        f"best model: lag={int(lag)}, window={int(window_size)}, top%={int(top_percent_key)} | "
        f"orig recon MSE={original_mse:.3e} | cf recon MSE={counterfactual_mse:.3e}",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_counterfactual_channel_timeseries(
    sample_id: str,
    label_name: str,
    target_label_name: str,
    channel_name: str,
    original_obs_channel: np.ndarray,
    cf_recon_channel: np.ndarray,
    lag: int,
    window_size: int,
    top_percent_key: int,
    counterfactual_mse: float,
    output_path: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4.8))
    x = np.arange(original_obs_channel.shape[0], dtype=np.int64)

    ax.plot(
        x,
        original_obs_channel,
        color="#1f77b4",
        linewidth=1.8,
        label=f"{channel_name} original",
    )
    ax.plot(
        x,
        cf_recon_channel,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label=f"{channel_name} counterfactual",
    )
    ax.set_xlabel("Minutes in observation window")
    ax.set_ylabel("Flux (pfu)")
    ax.set_yscale("log")
    positive_values = np.concatenate(
        [
            np.asarray(original_obs_channel, dtype=np.float64).ravel(),
            np.asarray(cf_recon_channel, dtype=np.float64).ravel(),
        ]
    )
    positive_values = positive_values[np.isfinite(positive_values) & (positive_values > 0)]
    if positive_values.size > 0:
        y_min = min(1e-2, float(np.min(positive_values)))
        y_max = float(np.max(positive_values))
        if y_max <= y_min:
            y_max = y_min * 10.0
        tick_candidates = np.array([0.1, 0.5, 1.0, y_max], dtype=np.float64)
        tick_candidates = tick_candidates[(tick_candidates >= y_min) & (tick_candidates <= y_max)]
        if tick_candidates.size == 0:
            tick_candidates = np.array([y_min, y_max], dtype=np.float64)
        ax.set_ylim(bottom=y_min, top=y_max)
        ax.set_yticks(np.unique(np.round(tick_candidates, 6)))
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle(
        f"Counterfactual Time Series | sample={sample_id} | channel={channel_name} | "
        f"{label_name} -> {target_label_name} | best model: lag={int(lag)}, "
        f"window={int(window_size)}, top%={int(top_percent_key)} | cf recon MSE={counterfactual_mse:.3e}",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate_counterfactual_reports(
    artifact: dict,
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    train_ids: list[str],
    classes: np.ndarray,
    channel_names: list[str],
    event_index: int,
    observation_window_size: int,
    output_dir: Path,
    run_stamp: str,
    random_state: int = 42,
    chosen_sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    if len(classes) != 2:
        raise ValueError("Experiment 16 counterfactual generation currently expects a binary classification task.")

    feature_bank = build_best_model_feature_bank(
        artifact=artifact,
        X_train=X_train,
        event_index=event_index,
        observation_window_size=observation_window_size,
    )
    X_train_flat = np.asarray(feature_bank["X_train_flat"], dtype=np.float32)
    selected_idx = np.asarray(artifact["selected_feature_indices"], dtype=np.int64)
    X_train_sel = X_train_flat[:, selected_idx]
    feature_names = list(artifact["selected_feature_names"])
    model = artifact["model"]

    chosen_examples = choose_training_examples_for_counterfactuals(
        X_train_sel=X_train_sel,
        y_train=y_train,
        train_ids=train_ids,
        model=model,
        classes=classes,
        num_per_class=2,
        chosen_sample_ids=chosen_sample_ids,
    )

    dice = make_dice_explainer(
        X_train_sel=X_train_sel,
        y_train=y_train,
        feature_names=feature_names,
        model=model,
    )

    rows: list[dict[str, float | int | str]] = []
    channels = int(feature_bank["channels"])
    n_slices = int(feature_bank["n_slices"])
    max_coeffs = int(feature_bank["max_coeffs"])
    fft_window_size = int(artifact["window_size"])
    lag = int(artifact["forecast_lag_min"])

    for example in chosen_examples:
        idx = int(example["train_index"])
        sample_id = str(example["sample_id"])
        label = int(example["label"])
        label_name = str(example["label_name"])
        original_full = X_train_flat[idx].astype(np.float64).copy()
        original_sel = X_train_sel[idx].astype(np.float64).copy()
        query_df = pd.DataFrame([original_sel], columns=feature_names)

        cf_generated = False
        error_message = ""
        changed_feature_count = 0
        counterfactual_mse = np.nan
        counterfactual_mae = np.nan
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
            changed_feature_count = int(np.sum(np.abs(cf_selected - original_sel) > 1e-9))

            cf_full = original_full.copy()
            cf_full[selected_idx] = cf_selected

            original_obs = extract_observation_window(
                X_train[idx],
                event_index=event_index,
                observation_window_size=observation_window_size,
                lag_minute=lag,
            )
            original_recon = reconstruct_observation_from_fft_features(
                original_full,
                channels=channels,
                n_slices=n_slices,
                max_coeffs=max_coeffs,
                fft_window_size=fft_window_size,
            )
            cf_recon = reconstruct_observation_from_fft_features(
                cf_full,
                channels=channels,
                n_slices=n_slices,
                max_coeffs=max_coeffs,
                fft_window_size=fft_window_size,
            )

            original_mse = float(np.mean((original_obs - original_recon) ** 2))
            counterfactual_mse = float(np.mean((original_obs - cf_recon) ** 2))
            counterfactual_mae = float(np.mean(np.abs(original_obs - cf_recon)))

            combined_plot_path = output_dir / "combined" / (
                f"experiment16_cf_sample_{sample_id}_lag{lag}_w{fft_window_size}_"
                f"top{int(artifact['top_percent_key'])}_{label_name}_to_{target_label_name}_{run_stamp}_combined.png"
            )
            plot_counterfactual_timeseries(
                sample_id=sample_id,
                label_name=label_name,
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
                channel_plot_path = output_dir / "channels" / (
                    f"experiment16_cf_sample_{sample_id}_lag{lag}_w{fft_window_size}_"
                    f"top{int(artifact['top_percent_key'])}_{channel_label}_{label_name}_"
                    f"to_{target_label_name}_{run_stamp}.png"
                )
                channel_mse = float(np.mean((original_obs[:, c] - cf_recon[:, c]) ** 2))
                plot_counterfactual_channel_timeseries(
                    sample_id=sample_id,
                    label_name=label_name,
                    target_label_name=target_label_name,
                    channel_name=channel_label,
                    original_obs_channel=original_obs[:, c],
                    cf_recon_channel=cf_recon[:, c],
                    lag=lag,
                    window_size=fft_window_size,
                    top_percent_key=int(artifact["top_percent_key"]),
                    counterfactual_mse=channel_mse,
                    output_path=channel_plot_path,
                )
            cf_generated = True
        except Exception as exc:
            cf_prediction = int(example["predicted_label"])
            original_obs = extract_observation_window(
                X_train[idx],
                event_index=event_index,
                observation_window_size=observation_window_size,
                lag_minute=lag,
            )
            original_recon = reconstruct_observation_from_fft_features(
                original_full,
                channels=channels,
                n_slices=n_slices,
                max_coeffs=max_coeffs,
                fft_window_size=fft_window_size,
            )
            original_mse = float(np.mean((original_obs - original_recon) ** 2))
            error_message = str(exc)

        rows.append(
            {
                "sample_id": sample_id,
                "train_index": idx,
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
                "changed_feature_count": int(changed_feature_count),
                "original_reconstruction_mse": float(original_mse),
                "counterfactual_reconstruction_mse": float(counterfactual_mse) if np.isfinite(counterfactual_mse) else np.nan,
                "counterfactual_reconstruction_mae": float(counterfactual_mae) if np.isfinite(counterfactual_mae) else np.nan,
                "error": error_message,
            }
        )

    return pd.DataFrame(rows)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"

    raw_dir = data_root / "raw"
    labels_file = data_root / "SEP_class_labels.csv"

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Missing data directory: '{raw_dir}'. "
            "Expected raw CSV files under data/raw/."
        )

    target_channels = ["p3_flux_ic", "p5_flux_ic", "p7_flux_ic", "long"]
    dataset = TimeSeriesDataset(
        data_dir=raw_dir,
        labels_file=labels_file,
        filename_col="File",
        label_col="Label",
        feature_cols=target_channels,
    )
    (X_train, y_train, train_ids), (X_val, y_val, _), (X_test, y_test, _) = get_splits_with_ids(dataset)

    event_onset_index = 720
    observation_window_size = 360
    fft_window_sizes = [45, 90, 180, 360]
    forecast_lags = [5, 15, 30, 60, 120]
    top_percents = [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]
    #chosen_counterfactual_files: list[str] = ["2003-07-05_23-06.csv","2010-05-05_10-37.csv",                                              "2003-10-28_00-15.csv","2011-11-25_23-25.csv"]
    chosen_counterfactual_files: list[str] =[]
    output_dir = data_root / "reports" / "experiment16"
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_run = find_best_cached_run(output_dir)

    if cached_run is not None:
        print(
            "Using cached best experiment16 model selected from existing results CSVs: "
            f"{cached_run['run_stamp']}."
        )
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_df = cached_run["results_df"]
        selection_df = cached_run["selection_df"]
        best_row = cached_run["best_row"]
        best_artifact = cached_run["best_artifact"]
        results_path = cached_run["results_path"]
        selection_path = cached_run["selection_path"]
        best_summary_path = cached_run["summary_path"]
        models_dir = output_dir / "saved_models"
        saved_model_paths = [cached_run["model_path"]]
        val_plot_by_lag_path = output_dir / f"experiment16_val_css_by_lag_{cached_run['run_stamp']}.png"
        test_plot_by_lag_path = output_dir / f"experiment16_test_css_by_lag_{cached_run['run_stamp']}.png"
        val_plot_by_window_path = output_dir / f"experiment16_val_css_by_window_{cached_run['run_stamp']}.png"
        test_plot_by_window_path = output_dir / f"experiment16_test_css_by_window_{cached_run['run_stamp']}.png"
        feature_count_plot_path = (
            output_dir / f"experiment16_features_required_cumimp_gt50_best_window_{cached_run['run_stamp']}.png"
        )
        best_rows_by_lag = select_best_per_lag(results_df)
        best_artifacts_by_lag = [
            load_artifact_for_row(output_dir, cached_run["run_stamp"], row)
            for _, row in best_rows_by_lag.iterrows()
        ]
    else:
        results_df, selection_df, model_artifacts = run_independent_window_top_percent_experiment(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            classes=dataset.classes_,
            channel_names=target_channels,
            fft_window_sizes=fft_window_sizes,
            forecast_lags=forecast_lags,
            top_percents=top_percents,
            event_index=event_onset_index,
            observation_window_size=observation_window_size,
            random_state=42,
        )

        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        results_path = output_dir / f"experiment16_window_independent_toppercent_results_{stamp}.csv"
        selection_path = output_dir / f"experiment16_window_independent_selection_details_{stamp}.csv"
        best_summary_path = output_dir / f"experiment16_best_model_summary_{stamp}.csv"
        results_df.to_csv(results_path, index=False)
        selection_df.to_csv(selection_path, index=False)

        models_dir = output_dir / "saved_models"
        saved_model_paths = save_model_artifacts(model_artifacts, models_dir, run_stamp=stamp)

        val_plot_by_lag_path = output_dir / f"experiment16_val_css_by_lag_{stamp}.png"
        plot_toppercent_performance_by_lag(results_df, val_plot_by_lag_path, metric_col="val_css")

        test_plot_by_lag_path = output_dir / f"experiment16_test_css_by_lag_{stamp}.png"
        plot_toppercent_performance_by_lag(results_df, test_plot_by_lag_path, metric_col="test_css")

        val_plot_by_window_path = output_dir / f"experiment16_val_css_by_window_{stamp}.png"
        plot_toppercent_performance_by_window(results_df, val_plot_by_window_path, metric_col="val_css")

        test_plot_by_window_path = output_dir / f"experiment16_test_css_by_window_{stamp}.png"
        plot_toppercent_performance_by_window(results_df, test_plot_by_window_path, metric_col="test_css")

        feature_count_plot_path = output_dir / f"experiment16_features_required_cumimp_gt50_best_window_{stamp}.png"
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

    counterfactual_plot_dir = output_dir / "counterfactual_timeseries" / stamp
    counterfactual_summary_path = output_dir / f"experiment16_counterfactual_summary_{stamp}.csv"
    counterfactual_frames: list[pd.DataFrame] = []
    for artifact in best_artifacts_by_lag:
        counterfactual_frames.append(
            generate_counterfactual_reports(
                artifact=artifact,
                X_train=X_train,
                y_train=y_train,
                train_ids=train_ids,
                classes=dataset.classes_,
                channel_names=target_channels,
                event_index=event_onset_index,
                observation_window_size=observation_window_size,
                output_dir=counterfactual_plot_dir,
                run_stamp=stamp,
                random_state=42,
                chosen_sample_ids=chosen_counterfactual_files,
            )
        )
    counterfactual_df = pd.concat(counterfactual_frames, axis=0, ignore_index=True) if counterfactual_frames else pd.DataFrame()
    counterfactual_df.to_csv(counterfactual_summary_path, index=False)

    print("\n" + "=" * 80)
    print("Experiment 16 (independent window top-percent models with DiCE counterfactuals) finished.")
    print(f"Run timestamp: {stamp}")
    print(f"Saved results CSV: {results_path}")
    print(f"Saved selection CSV: {selection_path}")
    print(f"Saved best-model summary CSV: {best_summary_path}")
    print(f"Saved counterfactual summary CSV: {counterfactual_summary_path}")
    print(f"Saved validation-by-lag plot: {val_plot_by_lag_path}")
    print(f"Saved test-by-lag plot: {test_plot_by_lag_path}")
    print(f"Saved validation-by-window plot: {val_plot_by_window_path}")
    print(f"Saved test-by-window plot: {test_plot_by_window_path}")
    print(f"Saved feature-count plot: {feature_count_plot_path}")
    print(f"Saved models: {len(saved_model_paths)} files in {models_dir}")
    print(f"Saved counterfactual time-series plots in: {counterfactual_plot_dir}")
    print(
        "Best model: "
        f"lag={int(best_row['forecast_lag_min'])}, "
        f"window={int(best_row['window_size'])}, "
        f"top%={int(round(float(best_row['top_percent']) * 100.0))}, "
        f"val_css={float(best_row['val_css']):.4f}, "
        f"test_css={float(best_row['test_css']):.4f}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
