"""Experiment 13: lag-wise concatenated FFT feature selection + RandomForest.

This script reuses Experiment 12 sweep logs (`experiment12_results_*.csv`) to
pick the best coefficient count per (lag, FFT-window) setup using validation
priority (CSS/TSS/HSS/F1, then smaller k).

After selecting best k per window for a given lag, it concatenates all selected
window features for that lag and trains/evaluates one RandomForest model.
It also plots validation curves grouped by lag using the logged sweep results.
"""

from __future__ import annotations

import os
import math
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.fft import rfft
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


class TimeSeriesDataset:
    def __init__(
        self,
        data_dir: Path,
        labels_file: Path,
        filename_col: str = "filename",
        label_col: str = "label",
        feature_cols: list[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.labels_file = Path(labels_file)
        self.filename_col = filename_col
        self.label_col = label_col
        self.feature_cols = feature_cols

        self.metadata = pd.read_csv(self.labels_file)
        self.metadata["full_path"] = self.metadata[self.filename_col].apply(
            lambda x: self.data_dir / (str(x) if str(x).endswith(".csv") else f"{x}.csv")
        )
        self.metadata = self.metadata[self.metadata["full_path"].apply(os.path.exists)].copy()
        self.metadata["event_time"] = self.metadata[self.filename_col].apply(self._parse_event_time)
        self.metadata = self.metadata[self.metadata["event_time"].notna()].copy()

        if len(self.metadata) == 0:
            raise ValueError("No valid files found for experiment.")

        self.label_encoder = LabelEncoder()
        self.metadata["encoded_label"] = self.label_encoder.fit_transform(self.metadata[self.label_col])
        self.classes_ = self.label_encoder.classes_

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def _parse_event_time(filename: str) -> pd.Timestamp :
        stem = Path(str(filename)).stem
        dt = pd.to_datetime(stem, format="%Y-%m-%d_%H-%M", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(stem, errors="coerce")
        return dt

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        row = self.metadata.iloc[idx]
        df = pd.read_csv(row["full_path"])

        if self.feature_cols:
            series = df[self.feature_cols].values
        else:
            series = df.values

        return series.astype(np.float32), int(row["encoded_label"])

    def get_splits(self):
        train_X, train_y = [], []
        val_X, val_y = [], []
        test_X, test_y = [], []

        for idx in range(len(self)):
            row = self.metadata.iloc[idx]
            year = int(row["event_time"].year)

            try:
                series, label = self[idx]
            except Exception as exc:
                print(f"Skipping sample index {idx}: {exc}")
                continue

            if year <= 1992:
                train_X.append(series)
                train_y.append(label)
            elif 1992 < year <= 2002:
                val_X.append(series)
                val_y.append(label)
            elif 2002 < year <= 2018:
                test_X.append(series)
                test_y.append(label)

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
        return (train_X, y_train), (val_X, y_val), (test_X, y_test)


def pick_positive_class(classes: np.ndarray) -> int:
    labels = [str(c).lower() for c in classes]
    priority_tokens = ["flare", "sep", "event", "positive", "yes"]

    for token in priority_tokens:
        for idx, label in enumerate(labels):
            if token in label:
                return idx

    return 1 if len(classes) > 1 else 0


def binary_confusion_parts(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> tuple[int, int, int, int]:
    yt = (y_true == positive_class).astype(int)
    yp = (y_pred == positive_class).astype(int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_solar_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> dict[str, float]:
    tn, fp, fn, tp = binary_confusion_parts(y_true, y_pred, positive_class)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tss = recall + specificity - 1.0

    hss_num = 2.0 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = hss_num / hss_den if hss_den != 0 else 0.0

    css = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"tss": tss, "hss": hss, "css": css}


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> dict[str, float]:
    solar = compute_solar_metrics(y_true, y_pred, positive_class)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, pos_label=positive_class, average="binary", zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=positive_class, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=positive_class, average="binary", zero_division=0),
        "tss": solar["tss"],
        "hss": solar["hss"],
        "css": solar["css"],
    }


def extract_observation_window(
    series: np.ndarray,
    event_index: int,
    observation_window_size: int,
    lag_minute: int,
) -> np.ndarray:
    """Extract [event-lag-window : event-lag] with optional right padding."""
    input_end = event_index - lag_minute
    input_start = input_end - observation_window_size

    if input_start < 0:
        raise ValueError(f"Invalid setup: input_start becomes negative ({input_start}).")

    timesteps, channels = series.shape
    if timesteps < input_end:
        pad = np.zeros((input_end - timesteps, channels), dtype=series.dtype)
        series = np.vstack([series, pad])

    return series[input_start:input_end, :]


def build_fft_features_all(
    X: list[np.ndarray],
    fft_window_size: int,
    event_index: int,
    observation_window_size: int,
    forecast_lag: int,
) -> tuple[np.ndarray, int, int, int]:
    """Build magnitude+phase FFT features using all coefficients.

    Output format: (N, C_fft, T_fft)
      - C_fft = original_channels * 2 (mag + phase)
      - T_fft = n_slices * max_coeffs
    """
    obs_len = observation_window_size
    if obs_len % fft_window_size != 0:
        raise ValueError(
            f"FFT window size {fft_window_size} must divide observation length {obs_len}."
        )

    n_slices = obs_len // fft_window_size
    transformed = []
    max_coeffs = None

    for series in X:
        obs = extract_observation_window(
            series,
            event_index=event_index,
            observation_window_size=observation_window_size,
            lag_minute=forecast_lag,
        )

        _, channels = obs.shape
        channel_tracks: list[list[float]] = [[] for _ in range(channels * 2)]

        for s in range(n_slices):
            a = s * fft_window_size
            b = a + fft_window_size
            chunk = obs[a:b, :]

            fft_vals = rfft(chunk, axis=0)
            mag = np.abs(fft_vals)
            phase = np.angle(fft_vals)

            if max_coeffs is None:
                max_coeffs = mag.shape[0]

            for c in range(channels):
                channel_tracks[c].extend(mag[:, c].tolist())
                channel_tracks[channels + c].extend(phase[:, c].tolist())

        transformed.append(np.asarray(channel_tracks, dtype=np.float32))

    if max_coeffs is None:
        raise ValueError("Unable to create FFT features: no valid samples.")

    X_all = np.asarray(transformed, dtype=np.float32)
    return X_all, max_coeffs, n_slices, channels


def select_top_k_coeffs(X_all: np.ndarray, max_coeffs: int, n_slices: int, n_coeffs: int | str) -> tuple[np.ndarray, int]:
    if n_coeffs == "all":
        k = max_coeffs
    else:
        k = min(int(n_coeffs), max_coeffs)

    # (N, C_fft, n_slices*max_coeffs) -> (N, C_fft, n_slices, max_coeffs)
    N, C_fft, _ = X_all.shape
    reshaped = X_all.reshape(N, C_fft, n_slices, max_coeffs)
    selected = reshaped[:, :, :, :k]
    return selected.reshape(N, C_fft, n_slices * k), k


def build_flat_features_for_config(
    X_train: list[np.ndarray],
    X_val: list[np.ndarray],
    X_test: list[np.ndarray],
    fft_window_size: int,
    forecast_lag: int,
    coeff_count: int,
    event_index: int,
    observation_window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    X_train_all, max_coeffs, n_slices, channels = build_fft_features_all(
        X_train,
        fft_window_size=fft_window_size,
        event_index=event_index,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
    )
    X_val_all, _, _, _ = build_fft_features_all(
        X_val,
        fft_window_size=fft_window_size,
        event_index=event_index,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
    )
    X_test_all, _, _, _ = build_fft_features_all(
        X_test,
        fft_window_size=fft_window_size,
        event_index=event_index,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
    )

    X_train_spec, k = select_top_k_coeffs(X_train_all, max_coeffs, n_slices, coeff_count)
    X_val_spec, _ = select_top_k_coeffs(X_val_all, max_coeffs, n_slices, coeff_count)
    X_test_spec, _ = select_top_k_coeffs(X_test_all, max_coeffs, n_slices, coeff_count)

    return (
        X_train_spec.reshape(X_train_spec.shape[0], -1),
        X_val_spec.reshape(X_val_spec.shape[0], -1),
        X_test_spec.reshape(X_test_spec.shape[0], -1),
        {
            "effective_coeffs": int(k),
            "max_available_coeffs": int(max_coeffs),
            "n_slices": int(n_slices),
            "channels": int(channels),
        },
    )


def build_flat_features_for_lag_concat(
    X_train: list[np.ndarray],
    X_val: list[np.ndarray],
    X_test: list[np.ndarray],
    fft_window_sizes: list[int],
    selected_coeffs_by_window: dict[int, int],
    forecast_lag: int,
    event_index: int,
    observation_window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, int]]]:
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    feature_meta: list[dict[str, int]] = []

    for fft_window_size in fft_window_sizes:
        window = int(fft_window_size)
        if window not in selected_coeffs_by_window:
            raise ValueError(f"Missing selected coefficient count for window={window}.")

        X_train_all, max_coeffs, n_slices, channels = build_fft_features_all(
            X_train,
            fft_window_size=window,
            event_index=event_index,
            observation_window_size=observation_window_size,
            forecast_lag=forecast_lag,
        )
        X_val_all, _, _, _ = build_fft_features_all(
            X_val,
            fft_window_size=window,
            event_index=event_index,
            observation_window_size=observation_window_size,
            forecast_lag=forecast_lag,
        )
        X_test_all, _, _, _ = build_fft_features_all(
            X_test,
            fft_window_size=window,
            event_index=event_index,
            observation_window_size=observation_window_size,
            forecast_lag=forecast_lag,
        )

        X_train_spec, k = select_top_k_coeffs(X_train_all, max_coeffs, n_slices, selected_coeffs_by_window[window])
        X_val_spec, _ = select_top_k_coeffs(X_val_all, max_coeffs, n_slices, selected_coeffs_by_window[window])
        X_test_spec, _ = select_top_k_coeffs(X_test_all, max_coeffs, n_slices, selected_coeffs_by_window[window])

        train_flat = X_train_spec.reshape(X_train_spec.shape[0], -1)
        val_flat = X_val_spec.reshape(X_val_spec.shape[0], -1)
        test_flat = X_test_spec.reshape(X_test_spec.shape[0], -1)
        train_parts.append(train_flat)
        val_parts.append(val_flat)
        test_parts.append(test_flat)

        feature_meta.append(
            {
                "window_size": window,
                "effective_coeffs": int(k),
                "max_available_coeffs": int(max_coeffs),
                "n_slices": int(n_slices),
                "channels": int(channels),
                "flat_dim": int(train_flat.shape[1]),
            }
        )

    if not train_parts:
        raise RuntimeError("No features were built for lag-level concatenation.")

    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(val_parts, axis=1),
        np.concatenate(test_parts, axis=1),
        feature_meta,
    )


def get_positive_class_probability(model, X: np.ndarray, positive_class: int) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is not None:
        idx_arr = np.where(classes == positive_class)[0]
        if len(idx_arr) > 0:
            return proba[:, int(idx_arr[0])]
    idx = min(max(int(positive_class), 0), proba.shape[1] - 1)
    return proba[:, idx]


def calibrate_prefit_sigmoid(base_model, X_cal: np.ndarray, y_cal: np.ndarray):
    if np.unique(y_cal).size < 2:
        return None

    try:
        calibrator = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=base_model, method="sigmoid", cv="prefit")
    calibrator.fit(X_cal, y_cal)
    return calibrator


def _safe_calibration_curve(y_true_bin: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_prob = np.unique(prob)
    n_bins = min(10, max(2, int(unique_prob.shape[0])))
    frac_pos, mean_pred = calibration_curve(y_true_bin, prob, n_bins=n_bins, strategy="quantile")
    return frac_pos, mean_pred


def plot_calibration_curves(
    y_val_bin: np.ndarray,
    y_test_bin: np.ndarray,
    val_prob_raw: np.ndarray,
    test_prob_raw: np.ndarray,
    val_prob_cal: np.ndarray,
    test_prob_cal: np.ndarray,
    output_path: Path,
    title_prefix: str = "",
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    prefix = f"{title_prefix} | " if str(title_prefix).strip() else ""

    for ax, split_name, y_bin, p_raw, p_cal in [
        (axes[0], "Validation", y_val_bin, val_prob_raw, val_prob_cal),
        (axes[1], "Test", y_test_bin, test_prob_raw, test_prob_cal),
    ]:
        raw_frac, raw_mean = _safe_calibration_curve(y_bin, p_raw)
        cal_frac, cal_mean = _safe_calibration_curve(y_bin, p_cal)

        brier_raw = brier_score_loss(y_bin, p_raw)
        brier_cal = brier_score_loss(y_bin, p_cal)

        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, label="perfect")
        ax.plot(raw_mean, raw_frac, color="#d62728", marker="o", linewidth=1.7, label="uncalibrated")
        ax.plot(cal_mean, cal_frac, color="#1f77b4", marker="o", linewidth=1.7, label="calibrated")
        ax.set_title(f"{prefix}{split_name} | Brier raw={brier_raw:.4f}, cal={brier_cal:.4f}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive frequency")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", frameon=False, fontsize=9, title="Calibration")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def apply_2k_coefficient_ticks(ax):
    """Display x ticks as 2k (mag+phase) while plotting against k."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{int(round(2 * x))}"))


def plot_metric_lineplots(results_df: pd.DataFrame, output_path: Path, split_prefix: str):
    metrics = [
        f"{split_prefix}_tss",
        f"{split_prefix}_hss",
        f"{split_prefix}_css",
        f"{split_prefix}_accuracy",
        f"{split_prefix}_f1",
        f"{split_prefix}_precision",
        f"{split_prefix}_recall",
    ]
    metric_ylim = {
        "tss": (-1.0, 1.0),
        "hss": (-1.0, 1.0),
        "css": (0.0, 1.0),
        "accuracy": (0.0, 1.0),
        "f1": (0.0, 1.0),
        "precision": (0.0, 1.0),
        "recall": (0.0, 1.0),
    }

    plot_df = results_df.copy()
    plot_df["window_size"] = plot_df["window_size"].astype(int)
    plot_df["forecast_lag_min"] = plot_df["forecast_lag_min"].astype(int)
    plot_df["window_lag"] = plot_df.apply(
        lambda r: f"W{int(r['window_size'])}_lag{int(r['forecast_lag_min'])}",
        axis=1,
    )

    setup_order = (
        plot_df[["window_size", "forecast_lag_min", "window_lag"]]
        .drop_duplicates()
        .sort_values(["window_size", "forecast_lag_min"])
        .itertuples(index=False, name=None)
    )
    setup_list = list(setup_order)
    colors = plt.cm.get_cmap("tab20", max(1, len(setup_list)))

    fig, axes = plt.subplots(2, 4, figsize=(26, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for color_idx, (window_size, lag, window_lag) in enumerate(setup_list):
            setup_df = (
                plot_df[
                    (plot_df["window_size"] == window_size)
                    & (plot_df["forecast_lag_min"] == lag)
                ]
                .sort_values("effective_coeffs")
            )

            ax.plot(
                setup_df["effective_coeffs"],
                setup_df[metric],
                marker="o",
                linewidth=1.5,
                markersize=3,
                color=colors(color_idx),
                label=window_lag,
            )
            if split_prefix == "test" and setup_df[metric].notna().any():
                peak_idx = setup_df[metric].idxmax()
                ax.scatter(
                    [setup_df.loc[peak_idx, "effective_coeffs"]],
                    [setup_df.loc[peak_idx, metric]],
                    color=colors(color_idx),
                    marker="X",
                    s=36,
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=5,
                )

        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("# coefficients (2k, mag+phase)")
        ax.set_ylabel("score")
        metric_key = metric.replace(f"{split_prefix}_", "", 1)
        y_min, y_max = metric_ylim.get(metric_key, (-1.0, 1.0))
        ax.set_ylim(y_min, y_max)
        apply_2k_coefficient_ticks(ax)
        ax.grid(True, alpha=0.25)

    if len(axes) > len(metrics):
        axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        if split_prefix == "test":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    linestyle="None",
                    markerfacecolor="#7f7f7f",
                    markeredgecolor="black",
                    markersize=7,
                    label="peak point",
                )
            )
            labels.append("peak point")
        fig.legend(
            handles,
            labels,
            loc="center left",
            ncol=1,
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(1.01, 0.5),
            title="Window/Lag",
            title_fontsize=9,
        )

    plt.tight_layout(rect=[0, 0, 0.84, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_consolidated_metrics_by_setup(
    results_df: pd.DataFrame,
    output_path: Path,
    split_prefix: str,
):
    metric_defs = [
        ("css", "CSS", "#1f77b4"),
        ("tss", "TSS", "#d62728"),
        ("hss", "HSS", "#2ca02c"),
        ("f1", "F1", "#ff7f0e"),
    ]

    plot_df = results_df.copy()
    plot_df["window_size"] = plot_df["window_size"].astype(int)
    plot_df["forecast_lag_min"] = plot_df["forecast_lag_min"].astype(int)

    setup_order = (
        plot_df[["window_size", "forecast_lag_min"]]
        .drop_duplicates()
        .sort_values(["window_size", "forecast_lag_min"])
        .itertuples(index=False, name=None)
    )
    setup_list = list(setup_order)
    if not setup_list:
        return

    n_cols = 5
    n_rows = int(math.ceil(len(setup_list) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    flat_axes = axes.flatten()

    for idx, (window_size, lag) in enumerate(setup_list):
        ax = flat_axes[idx]
        setup_df = (
            plot_df[
                (plot_df["window_size"] == int(window_size))
                & (plot_df["forecast_lag_min"] == int(lag))
            ]
            .sort_values("effective_coeffs")
        )

        for metric_key, metric_label, color in metric_defs:
            col = f"{split_prefix}_{metric_key}"
            ax.plot(
                setup_df["effective_coeffs"],
                setup_df[col],
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=2.6,
                label=metric_label,
            )

        ax.set_title(f"W{int(window_size)}_lag{int(lag)}")
        ax.set_xlabel("# coefficients (2k, mag+phase)")
        ax.set_ylabel("score")
        ax.set_ylim(-1.0, 1.0)
        apply_2k_coefficient_ticks(ax)
        ax.grid(True, alpha=0.25)

    for ax in flat_axes[len(setup_list):]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            ncol=1,
            frameon=False,
            fontsize=11,
            bbox_to_anchor=(1.01, 0.5),
            title="Metrics",
            title_fontsize=12,
        )

    fig.suptitle(f"{split_prefix.upper()} consolidated metrics per setup", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 0.86, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_validation_elbow_curves(
    results_df: pd.DataFrame,
    setup_summary_df: pd.DataFrame,
    output_path: Path,
):
    if len(results_df) == 0 or len(setup_summary_df) == 0:
        return

    setup_keys = (
        setup_summary_df[["window_size", "forecast_lag_min"]]
        .drop_duplicates()
        .sort_values(["window_size", "forecast_lag_min"])
        .itertuples(index=False, name=None)
    )
    setup_list = list(setup_keys)
    n_setups = len(setup_list)
    n_cols = 5
    n_rows = int(math.ceil(n_setups / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.flatten()

    for ax_idx, (window_size, lag) in enumerate(setup_list):
        ax = flat_axes[ax_idx]
        curve_df = (
            results_df[
                (results_df["window_size"].astype(int) == int(window_size))
                & (results_df["forecast_lag_min"].astype(int) == int(lag))
            ]
            .sort_values("effective_coeffs")
        )
        summary_row = setup_summary_df[
            (setup_summary_df["window_size"].astype(int) == int(window_size))
            & (setup_summary_df["forecast_lag_min"].astype(int) == int(lag))
        ].iloc[0]

        ax.plot(
            curve_df["effective_coeffs"],
            curve_df["val_css"],
            color="#1f77b4",
            linewidth=1.8,
            label="val_css",
        )
        ax.plot(
            curve_df["effective_coeffs"],
            curve_df["test_css"],
            color="#7f7f7f",
            linewidth=1.2,
            alpha=0.75,
            linestyle="-",
            label="test_css",
        )

        best_k = int(summary_row["best_val_k"])
        elbow_k = int(summary_row["elbow_k"])
        cv_k = int(summary_row["cv_selected_k"])
        test_peak_k = int(curve_df.loc[curve_df["test_css"].idxmax(), "effective_coeffs"])

        best_y = float(curve_df[curve_df["effective_coeffs"] == best_k]["val_css"].iloc[0])
        elbow_y = float(curve_df[curve_df["effective_coeffs"] == elbow_k]["val_css"].iloc[0])
        cv_y = float(curve_df[curve_df["effective_coeffs"] == cv_k]["val_css"].iloc[0])
        test_peak_y = float(curve_df[curve_df["effective_coeffs"] == test_peak_k]["test_css"].iloc[0])

        ax.scatter([best_k], [best_y], color="#d62728", s=55, marker="o", label="best")
        ax.scatter([elbow_k], [elbow_y], color="#2ca02c", s=60, marker="D", label="elbow")
        ax.scatter([cv_k], [cv_y], color="#ff7f0e", s=90, marker="*", label="cv")
        ax.scatter([test_peak_k], [test_peak_y], color="#9467bd", s=65, marker="X", label="test_peak")

        ax.set_title(f"W{int(window_size)}_lag{int(lag)}")
        ax.set_xlabel("# coefficients (2k, mag+phase)")
        ax.set_ylabel("CSS")
        ax.set_ylim(0.0, 1.0)
        apply_2k_coefficient_ticks(ax)
        ax.grid(True, alpha=0.25)

    for ax in flat_axes[n_setups:]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            ncol=1,
            frameon=False,
            bbox_to_anchor=(1.01, 0.5),
            fontsize=10,
            title="Curves & Picks",
            title_fontsize=11,
        )

    plt.tight_layout(rect=[0, 0, 0.86, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_validation_curves_by_lag(
    results_df: pd.DataFrame,
    setup_summary_df: pd.DataFrame,
    output_path: Path,
):
    if len(results_df) == 0:
        return

    lags = sorted(results_df["forecast_lag_min"].astype(int).unique().tolist())
    if not lags:
        return

    n_cols = min(3, len(lags))
    n_rows = int(math.ceil(len(lags) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8 * n_cols, 5 * n_rows),
        squeeze=False,
        sharey=True,
    )
    flat_axes = axes.flatten()

    for lag_idx, lag in enumerate(lags):
        ax = flat_axes[lag_idx]
        lag_df = results_df[results_df["forecast_lag_min"].astype(int) == int(lag)].copy()
        window_sizes = sorted(lag_df["window_size"].astype(int).unique().tolist())
        colors = plt.cm.get_cmap("tab10", max(1, len(window_sizes)))

        for color_idx, window_size in enumerate(window_sizes):
            curve_df = (
                lag_df[lag_df["window_size"].astype(int) == int(window_size)]
                .sort_values("effective_coeffs")
            )
            ax.plot(
                curve_df["effective_coeffs"],
                curve_df["val_css"],
                color=colors(color_idx),
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=f"W{int(window_size)}",
            )

            selected_row = setup_summary_df[
                (setup_summary_df["forecast_lag_min"].astype(int) == int(lag))
                & (setup_summary_df["window_size"].astype(int) == int(window_size))
            ]
            if len(selected_row) > 0:
                selected_k = int(selected_row.iloc[0]["best_val_k"])
                selected_curve = curve_df[curve_df["effective_coeffs"].astype(int) == selected_k]
                if len(selected_curve) > 0:
                    ax.scatter(
                        [selected_k],
                        [float(selected_curve.iloc[0]["val_css"])],
                        color=colors(color_idx),
                        marker="X",
                        s=45,
                        edgecolor="black",
                        linewidth=0.3,
                        zorder=5,
                    )

        ax.set_title(f"Lag {int(lag)} min")
        ax.set_xlabel("# coefficients (2k, mag+phase)")
        ax.set_ylabel("Validation CSS")
        ax.set_ylim(0.0, 1.0)
        apply_2k_coefficient_ticks(ax)
        ax.grid(True, alpha=0.25)

    for ax in flat_axes[len(lags):]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="X",
                linestyle="None",
                markerfacecolor="#7f7f7f",
                markeredgecolor="black",
                markersize=7,
                label="selected k",
            )
        )
        labels.append("selected k")
        fig.legend(
            handles,
            labels,
            loc="center left",
            ncol=1,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(1.01, 0.5),
            title="Windows",
            title_fontsize=11,
        )

    fig.suptitle("Validation curves by lag", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 0.86, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_lag_level_validation_by_lag(
    lag_results_df: pd.DataFrame,
    output_path: Path,
):
    if len(lag_results_df) == 0:
        return

    plot_df = lag_results_df.copy()
    plot_df["forecast_lag_min"] = plot_df["forecast_lag_min"].astype(int)
    plot_df = plot_df.sort_values("forecast_lag_min")

    x = plot_df["forecast_lag_min"].values
    val_css = plot_df["val_css"].values
    test_css = plot_df["test_css"].values

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(x, val_css, color="#1f77b4", marker="o", linewidth=2.0, label="val_css (concatenated)")
    ax.plot(x, test_css, color="#7f7f7f", marker="o", linewidth=1.5, linestyle="--", label="test_css (concatenated)")

    ax.set_title("Lag-level concatenated model performance")
    ax.set_xlabel("Forecast lag (minutes)")
    ax.set_ylabel("CSS")
    ax.set_xticks(x)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_feature_spans(feature_meta: list[dict[str, int]]) -> list[dict[str, int]]:
    spans: list[dict[str, int]] = []
    start = 0
    for meta in feature_meta:
        width = int(meta.get("flat_dim", 0))
        end = start + width
        spans.append(
            {
                "window_size": int(meta["window_size"]),
                "start": int(start),
                "end": int(end),
            }
        )
        start = end
    return spans


def locate_feature_window(feature_index: int, spans: list[dict[str, int]]) -> tuple[int, int]:
    for span in spans:
        if int(span["start"]) <= int(feature_index) < int(span["end"]):
            return int(span["window_size"]), int(feature_index - int(span["start"]))
    return -1, int(feature_index)


def decode_local_feature_components(
    local_idx: int,
    window_meta: dict[str, int],
    channel_names: list[str] | None = None,
) -> dict[str, int | str]:
    channels = int(window_meta["channels"])
    n_slices = int(window_meta["n_slices"])
    k = int(window_meta["effective_coeffs"])
    span_per_channel = n_slices * k

    if span_per_channel <= 0:
        return {
            "mag_phase": "unknown",
            "channel_index": -1,
            "channel_name": "unknown",
            "slice_index": -1,
            "coeff_index": -1,
        }

    channel_fft = int(local_idx) // span_per_channel
    rem = int(local_idx) % span_per_channel
    slice_index = rem // k
    coeff_index = rem % k

    if channel_fft < channels:
        mag_phase = "mag"
        channel_index = channel_fft
    else:
        mag_phase = "phase"
        channel_index = channel_fft - channels

    if channel_names and 0 <= channel_index < len(channel_names):
        channel_name = str(channel_names[channel_index])
    else:
        channel_name = f"ch{channel_index}"

    return {
        "mag_phase": mag_phase,
        "channel_index": int(channel_index),
        "channel_name": channel_name,
        "slice_index": int(slice_index),
        "coeff_index": int(coeff_index),
    }


def build_lag_feature_names(
    feature_meta: list[dict[str, int]],
    channel_names: list[str] | None = None,
) -> list[str]:
    names: list[str] = []
    for meta in feature_meta:
        window = int(meta["window_size"])
        channels = int(meta["channels"])
        n_slices = int(meta["n_slices"])
        k = int(meta["effective_coeffs"])

        for channel_fft in range(2 * channels):
            if channel_fft < channels:
                mag_phase = "mag"
                channel_index = channel_fft
            else:
                mag_phase = "phase"
                channel_index = channel_fft - channels

            if channel_names and 0 <= channel_index < len(channel_names):
                channel_name = str(channel_names[channel_index])
            else:
                channel_name = f"ch{channel_index}"

            for s_idx in range(n_slices):
                for k_idx in range(k):
                    names.append(
                        f"W{window}_{mag_phase}_{channel_name}_s{s_idx + 1}_k{k_idx + 1}"
                    )
    return names


def _load_sep_cfe_dice_api(sep_project_root: Path):
    if not sep_project_root.exists():
        raise FileNotFoundError(f"SEP project root not found: {sep_project_root}")

    root_str = str(sep_project_root)
    if root_str not in sys.path:
        sys.path.append(root_str)

    from src.SEP_CFE_DiCE import (  # type: ignore
        build_constrained_explainer,
        extract_counterfactual_dfs,
        generate_counterfactuals as dice_generate_counterfactuals,
    )

    return build_constrained_explainer, dice_generate_counterfactuals, extract_counterfactual_dfs


def plot_feature_importance_for_lag(
    feature_importances: np.ndarray,
    feature_meta: list[dict[str, int]],
    lag: int,
    output_path: Path,
    top_n: int = 25,
    channel_names: list[str] | None = None,
):
    if feature_importances.size == 0:
        return

    spans = build_feature_spans(feature_meta)
    top_k = min(int(top_n), int(feature_importances.size))
    top_idx = np.argsort(feature_importances)[-top_k:][::-1]
    top_vals = feature_importances[top_idx]
    top_labels: list[str] = []
    meta_by_window = {int(m["window_size"]): m for m in feature_meta}
    for idx in top_idx:
        window_size, local_idx = locate_feature_window(int(idx), spans)
        if window_size > 0:
            window_meta = meta_by_window.get(int(window_size), {})
            decoded = decode_local_feature_components(
                int(local_idx),
                window_meta,
                channel_names=channel_names,
            )
            top_labels.append(
                "W"
                f"{window_size}:"
                f"{decoded['mag_phase']}:"
                f"{decoded['channel_name']}:"
                f"s{int(decoded['slice_index']) + 1}:"
                f"idx{int(decoded['coeff_index']) + 1}"
            )
        else:
            top_labels.append(f"f{int(idx)}")

    window_labels: list[str] = []
    window_vals: list[float] = []
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        window_size = int(span["window_size"])
        window_labels.append(f"W{window_size}")
        window_vals.append(float(feature_importances[start:end].sum()))

    coeff_labels = [f"W{int(m['window_size'])}" for m in feature_meta]
    coeff_vals = [2 * int(m["effective_coeffs"]) for m in feature_meta]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].barh(range(top_k), top_vals[::-1], color="#1f77b4")
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels(top_labels[::-1], fontsize=8)
    axes[0].set_xlabel("importance")
    axes[0].set_title(f"Lag {int(lag)}: top {top_k} features")
    axes[0].grid(True, axis="x", alpha=0.25)

    bars1 = axes[1].bar(window_labels, window_vals, color="#2ca02c")
    axes[1].set_xlabel("window")
    axes[1].set_ylabel("summed importance")
    axes[1].set_title(f"Lag {int(lag)}: importance by window")
    axes[1].grid(True, axis="y", alpha=0.25)
    for bar in bars1:
        height = float(bar.get_height())
        x = float(bar.get_x() + bar.get_width() / 2.0)
        axes[1].text(
            x,
            height,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    bars2 = axes[2].bar(coeff_labels, coeff_vals, color="#ff7f0e")
    axes[2].set_xlabel("window")
    axes[2].set_ylabel("selected coefficients (2k, mag+phase)")
    axes[2].set_title(f"Lag {int(lag)}: selected coefficients per window (2k, mag+phase)")
    axes[2].grid(True, axis="y", alpha=0.25)
    for bar in bars2:
        height = float(bar.get_height())
        x = float(bar.get_x() + bar.get_width() / 2.0)
        axes[2].text(
            x,
            height,
            f"{int(round(height))}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_selected_coeffs_by_window_for_lag(
    setup_summary_df: pd.DataFrame,
    lag: int,
    fft_window_sizes: list[int],
) -> dict[int, int]:
    lag_selection_df = (
        setup_summary_df[setup_summary_df["forecast_lag_min"].astype(int) == int(lag)]
        .sort_values("window_size")
    )
    selected: dict[int, int] = {}
    for window in fft_window_sizes:
        row = lag_selection_df[lag_selection_df["window_size"].astype(int) == int(window)]
        if len(row) == 0:
            continue
        selected[int(window)] = int(row.iloc[0]["best_val_k"])
    return selected


def run_calibration_and_feature_importance_for_all_lags(
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_val: list[np.ndarray],
    y_val: np.ndarray,
    X_test: list[np.ndarray],
    y_test: np.ndarray,
    classes: np.ndarray,
    fft_window_sizes: list[int],
    forecast_lags: list[int],
    setup_summary_df: pd.DataFrame,
    event_index: int,
    observation_window_size: int,
    output_dir: Path,
    stamp: str,
    random_state: int = 42,
    top_n_features: int = 25,
    channel_names: list[str] | None = None,
    save_models_dir: Path | None = None,
    enable_constrained_dice: bool = True,
    sep_project_root: Path | None = None,
    counterfactuals_output_dir: Path | None = None,
    counterfactuals_per_query: int = 3,
    max_queries_per_lag: int = 5,
    outcome_name: str = "Event_Y_N",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive_class = pick_positive_class(classes)
    y_val_bin = (y_val == positive_class).astype(int)
    y_test_bin = (y_test == positive_class).astype(int)

    calibration_rows: list[dict[str, float | int | str]] = []
    feature_rows: list[dict[str, float | int | str]] = []
    counterfactual_rows: list[dict[str, float | int | str]] = []

    build_constrained_explainer = None
    dice_generate_counterfactuals = None
    extract_counterfactual_dfs = None
    if enable_constrained_dice:
        if sep_project_root is None:
            print("ConstrainedDiCE disabled: missing sep_project_root.")
        else:
            try:
                (
                    build_constrained_explainer,
                    dice_generate_counterfactuals,
                    extract_counterfactual_dfs,
                ) = _load_sep_cfe_dice_api(sep_project_root)
            except Exception as exc:
                print(f"ConstrainedDiCE disabled: {exc}")

    for lag in forecast_lags:
        selected_coeffs_by_window = build_selected_coeffs_by_window_for_lag(
            setup_summary_df=setup_summary_df,
            lag=int(lag),
            fft_window_sizes=fft_window_sizes,
        )
        if not selected_coeffs_by_window:
            print(f"Skipping lag={lag}: no selected coefficients for calibration/importance.")
            continue

        selected_windows = sorted(selected_coeffs_by_window.keys())
        X_train_flat, X_val_flat, X_test_flat, feature_meta = build_flat_features_for_lag_concat(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            fft_window_sizes=selected_windows,
            selected_coeffs_by_window=selected_coeffs_by_window,
            forecast_lag=int(lag),
            event_index=event_index,
            observation_window_size=observation_window_size,
        )
        feature_names = build_lag_feature_names(feature_meta, channel_names=channel_names)
        if len(feature_names) != X_train_flat.shape[1]:
            raise ValueError(
                f"Feature-name length mismatch for lag={lag}: "
                f"{len(feature_names)} names vs {X_train_flat.shape[1]} features."
            )

        train_feature_df = pd.DataFrame(X_train_flat, columns=feature_names)
        val_feature_df = pd.DataFrame(X_val_flat, columns=feature_names)
        test_feature_df = pd.DataFrame(X_test_flat, columns=feature_names)

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state + int(lag),
            n_jobs=-1,
        )
        model.fit(train_feature_df, y_train)

        model_path = ""
        if save_models_dir is not None:
            save_models_dir.mkdir(parents=True, exist_ok=True)
            model_file_path = save_models_dir / f"experiment14_lag{int(lag)}_rf_model_{stamp}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "forecast_lag_min": int(lag),
                    "feature_names": feature_names,
                    "feature_meta": feature_meta,
                    "selected_coeffs_by_window": selected_coeffs_by_window,
                    "positive_class": int(positive_class),
                    "channel_names": list(channel_names or []),
                    "event_index": int(event_index),
                    "observation_window_size": int(observation_window_size),
                },
                model_file_path,
            )
            model_path = str(model_file_path)

        val_pred = model.predict(val_feature_df)
        val_prob_raw = get_positive_class_probability(model, val_feature_df, positive_class)
        test_prob_raw = get_positive_class_probability(model, test_feature_df, positive_class)

        calibrator = None
        calibration_status = "calibrated_sigmoid_prefit"
        try:
            calibrator = calibrate_prefit_sigmoid(model, val_feature_df, y_val)
            if calibrator is None:
                calibration_status = "skipped_single_class_validation"
        except Exception as exc:
            calibration_status = f"calibration_failed: {exc}"
            print(f"Calibration warning for lag={lag}: {exc}")

        if calibrator is not None:
            val_prob_cal = get_positive_class_probability(calibrator, val_feature_df, positive_class)
            test_prob_cal = get_positive_class_probability(calibrator, test_feature_df, positive_class)
        else:
            val_prob_cal = val_prob_raw.copy()
            test_prob_cal = test_prob_raw.copy()

        calibration_plot_path = output_dir / f"experiment14_lag{int(lag)}_calibration_{stamp}.png"
        plot_calibration_curves(
            y_val_bin=y_val_bin,
            y_test_bin=y_test_bin,
            val_prob_raw=val_prob_raw,
            test_prob_raw=test_prob_raw,
            val_prob_cal=val_prob_cal,
            test_prob_cal=test_prob_cal,
            output_path=calibration_plot_path,
            title_prefix=f"Lag {int(lag)} min",
        )

        coeff_tokens = [f"W{m['window_size']}:{m['effective_coeffs']}" for m in feature_meta]
        calibration_rows.append(
            {
                "forecast_lag_min": int(lag),
                "windows_used": ",".join(str(m["window_size"]) for m in feature_meta),
                "selected_coefficients": ",".join(coeff_tokens),
                "total_input_dim": int(X_train_flat.shape[1]),
                "calibration_status": calibration_status,
                "val_brier_raw": float(brier_score_loss(y_val_bin, val_prob_raw)),
                "val_brier_calibrated": float(brier_score_loss(y_val_bin, val_prob_cal)),
                "test_brier_raw": float(brier_score_loss(y_test_bin, test_prob_raw)),
                "test_brier_calibrated": float(brier_score_loss(y_test_bin, test_prob_cal)),
                "calibration_plot_path": str(calibration_plot_path),
                "model_path": model_path,
            }
        )

        feature_importances = np.asarray(model.feature_importances_, dtype=np.float64)
        fi_plot_path = output_dir / f"experiment14_lag{int(lag)}_feature_importance_{stamp}.png"
        plot_feature_importance_for_lag(
            feature_importances=feature_importances,
            feature_meta=feature_meta,
            lag=int(lag),
            output_path=fi_plot_path,
            top_n=top_n_features,
            channel_names=channel_names,
        )

        meta_by_window = {int(m["window_size"]): m for m in feature_meta}
        spans = build_feature_spans(feature_meta)
        for idx, value in enumerate(feature_importances):
            window_size, local_idx = locate_feature_window(int(idx), spans)
            decoded = {
                "mag_phase": "unknown",
                "channel_index": -1,
                "channel_name": "unknown",
                "slice_index": -1,
                "coeff_index": -1,
            }
            if window_size in meta_by_window:
                decoded = decode_local_feature_components(
                    int(local_idx),
                    meta_by_window[int(window_size)],
                    channel_names=channel_names,
                )
            feature_rows.append(
                {
                    "forecast_lag_min": int(lag),
                    "feature_index": int(idx),
                    "feature_local_index": int(local_idx),
                    "window_size": int(window_size),
                    "mag_phase": str(decoded["mag_phase"]),
                    "channel_index": int(decoded["channel_index"]),
                    "channel_name": str(decoded["channel_name"]),
                    "slice_index": int(decoded["slice_index"]),
                    "coeff_index": int(decoded["coeff_index"]),
                    "importance": float(value),
                    "feature_importance_plot_path": str(fi_plot_path),
                }
            )

        cf_status = "disabled"
        cf_output_path = ""
        cf_rows_generated = 0
        queries_used = 0
        if (
            build_constrained_explainer is not None
            and dice_generate_counterfactuals is not None
            and extract_counterfactual_dfs is not None
        ):
            try:
                candidate_idx = np.where(val_pred != positive_class)[0]
                if candidate_idx.size == 0:
                    candidate_idx = np.arange(val_feature_df.shape[0])
                query_idx = candidate_idx[: max(0, int(max_queries_per_lag))]
                queries_used = int(query_idx.shape[0])

                if queries_used == 0:
                    cf_status = "skipped_no_queries"
                else:
                    train_dice_df = train_feature_df.copy()
                    train_dice_df[outcome_name] = y_train
                    query_df = val_feature_df.iloc[query_idx].copy()

                    explainer, _ = build_constrained_explainer(
                        dataframe=train_dice_df,
                        model=model,
                        outcome_name=outcome_name,
                        constraints=None,
                        continuous_features=feature_names,
                        backend="sklearn",
                        model_type="classifier",
                    )

                    cf_obj = dice_generate_counterfactuals(
                        explainer=explainer,
                        query_instances=query_df,
                        total_cfs=int(counterfactuals_per_query),
                        desired_class=int(positive_class),
                        proximity_weight=0.2,
                        sparsity_weight=0.2,
                        diversity_weight=5.0,
                    )

                    per_query_cf_dfs = extract_counterfactual_dfs(cf_obj)
                    lag_cf_tables: list[pd.DataFrame] = []
                    for q_i, cf_df in enumerate(per_query_cf_dfs):
                        if cf_df is None or len(cf_df) == 0:
                            continue
                        src_idx = int(query_idx[q_i]) if q_i < len(query_idx) else -1
                        augmented = cf_df.copy()
                        augmented.insert(0, "forecast_lag_min", int(lag))
                        augmented.insert(1, "query_row_in_val", src_idx)
                        if src_idx >= 0:
                            augmented.insert(2, "query_true_label", int(y_val[src_idx]))
                            augmented.insert(3, "query_pred_label", int(val_pred[src_idx]))
                        lag_cf_tables.append(augmented)

                    if lag_cf_tables:
                        if counterfactuals_output_dir is None:
                            counterfactuals_output_dir = output_dir / "counterfactuals"
                        counterfactuals_output_dir.mkdir(parents=True, exist_ok=True)
                        lag_cf_df = pd.concat(lag_cf_tables, ignore_index=True)
                        lag_cf_path = (
                            counterfactuals_output_dir
                            / f"experiment14_lag{int(lag)}_counterfactuals_{stamp}.csv"
                        )
                        lag_cf_df.to_csv(lag_cf_path, index=False)
                        cf_output_path = str(lag_cf_path)
                        cf_rows_generated = int(len(lag_cf_df))
                        cf_status = "ok"
                    else:
                        cf_status = "generated_empty"
            except Exception as exc:
                cf_status = f"failed: {exc}"

        counterfactual_rows.append(
            {
                "forecast_lag_min": int(lag),
                "queries_used": int(queries_used),
                "counterfactuals_per_query": int(counterfactuals_per_query),
                "counterfactual_rows_generated": int(cf_rows_generated),
                "counterfactual_status": cf_status,
                "counterfactual_output_path": cf_output_path,
                "model_path": model_path,
            }
        )

    calibration_df = pd.DataFrame(calibration_rows)
    feature_df = pd.DataFrame(feature_rows)
    counterfactual_df = pd.DataFrame(counterfactual_rows)
    if len(feature_df) > 0:
        feature_df["rank_within_lag"] = (
            feature_df.groupby("forecast_lag_min")["importance"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
    return calibration_df, feature_df, counterfactual_df


def build_dynamic_coeff_options(max_coeffs: int) -> list[int]:
    return list(range(1, int(max_coeffs) + 1))


def sort_by_validation_priority(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "val_css",
            "val_tss",
            "val_hss",
            "val_f1",
            "effective_coeffs",
            "test_css",
            "test_tss",
            "test_hss",
            "test_f1",
        ],
        ascending=[False, False, False, False, True, False, False, False, False],
    )


def select_best(results_df: pd.DataFrame) -> pd.Series:
    return sort_by_validation_priority(results_df).iloc[0]


def sort_lag_results_by_validation_priority(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=[
            "val_css",
            "val_tss",
            "val_hss",
            "val_f1",
            "test_css",
            "test_tss",
            "test_hss",
            "test_f1",
        ],
        ascending=[False, False, False, False, False, False, False, False],
    )


def select_best_lag_result(lag_results_df: pd.DataFrame) -> pd.Series:
    return sort_lag_results_by_validation_priority(lag_results_df).iloc[0]


def find_latest_results_log(results_dir: Path, pattern: str = "experiment12_results_*.csv") -> Path:
    candidates = sorted(results_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No results file matched '{pattern}' in '{results_dir}'."
        )
    return candidates[-1]


def load_results_log(results_csv_path: Path) -> pd.DataFrame:
    required_cols = [
        "window_size",
        "forecast_lag_min",
        "effective_coeffs",
        "max_available_coeffs",
        "val_accuracy",
        "val_css",
        "val_tss",
        "val_hss",
        "val_f1",
        "val_precision",
        "val_recall",
        "test_accuracy",
        "test_css",
        "test_tss",
        "test_hss",
        "test_f1",
        "test_precision",
        "test_recall",
    ]
    df = pd.read_csv(results_csv_path)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Results log missing required columns: {missing}. "
            f"File: {results_csv_path}"
        )

    for col in ["window_size", "forecast_lag_min", "effective_coeffs", "max_available_coeffs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["window_size", "forecast_lag_min", "effective_coeffs"]).copy()
    df["window_size"] = df["window_size"].astype(int)
    df["forecast_lag_min"] = df["forecast_lag_min"].astype(int)
    df["effective_coeffs"] = df["effective_coeffs"].astype(int)
    df["max_available_coeffs"] = df["max_available_coeffs"].fillna(0).astype(int)
    return df


def build_setup_summary_from_logs(
    results_df: pd.DataFrame,
    elbow_tolerance: float = 0.005,
) -> pd.DataFrame:
    setup_rows: list[dict[str, float | int | str]] = []

    grouped = results_df.groupby(["window_size", "forecast_lag_min"], sort=True)
    for (window_size, lag), setup_df in grouped:
        best_row = sort_by_validation_priority(setup_df).iloc[0]
        elbow_row = find_elbow_row(
            setup_df,
            metric_col="val_css",
            tolerance=elbow_tolerance,
        )

        input_start = np.nan
        input_end = np.nan
        if "input_start" in setup_df.columns and pd.notna(best_row["input_start"]):
            input_start = int(best_row["input_start"])
        if "input_end" in setup_df.columns and pd.notna(best_row["input_end"]):
            input_end = int(best_row["input_end"])
        best_k = int(best_row["effective_coeffs"])
        elbow_k = int(elbow_row["effective_coeffs"])
        best_val_css = float(best_row["val_css"])

        setup_rows.append(
            {
                "window_size": int(window_size),
                "forecast_lag_min": int(lag),
                "input_start": input_start,
                "input_end": input_end,
                "max_available_coeffs": int(setup_df["max_available_coeffs"].max()),
                "best_val_k": best_k,
                "best_val_css": best_val_css,
                "best_val_tss": float(best_row["val_tss"]),
                "best_val_hss": float(best_row["val_hss"]),
                "best_val_f1": float(best_row["val_f1"]),
                "elbow_k": elbow_k,
                "elbow_val_css": float(elbow_row["val_css"]),
                "elbow_tolerance": float(elbow_tolerance),
                "elbow_threshold_css": float(best_val_css - elbow_tolerance),
                # CV metrics are unavailable when selecting purely from logs.
                "cv_selected_k": best_k,
                "cv_css_mean": np.nan,
                "cv_tss_mean": np.nan,
                "cv_hss_mean": np.nan,
                "cv_f1_mean": np.nan,
                "cv_folds_used": np.nan,
                "cv_candidates": "",
                "selection_source": "experiment12_results_log",
            }
        )

    if not setup_rows:
        raise RuntimeError("No setup summary rows could be derived from results logs.")
    return pd.DataFrame(setup_rows)


def run_lag_concatenated_from_logs(
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_val: list[np.ndarray],
    y_val: np.ndarray,
    X_test: list[np.ndarray],
    y_test: np.ndarray,
    classes: np.ndarray,
    fft_window_sizes: list[int],
    forecast_lags: list[int],
    setup_summary_df: pd.DataFrame,
    event_index: int,
    observation_window_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    positive_class = pick_positive_class(classes)
    lag_rows: list[dict[str, float | int | str]] = []

    for lag in forecast_lags:
        input_end = event_index - int(lag)
        input_start = input_end - observation_window_size
        lag_selection_df = (
            setup_summary_df[setup_summary_df["forecast_lag_min"].astype(int) == int(lag)]
            .sort_values("window_size")
        )
        if len(lag_selection_df) == 0:
            print(f"Skipping lag={lag}: no setup selections found in logs.")
            continue

        selected_coeffs_by_window: dict[int, int] = {}
        for window in fft_window_sizes:
            window_row = lag_selection_df[lag_selection_df["window_size"].astype(int) == int(window)]
            if len(window_row) == 0:
                continue
            selected_coeffs_by_window[int(window)] = int(window_row.iloc[0]["best_val_k"])

        if not selected_coeffs_by_window:
            print(f"Skipping lag={lag}: no overlapping window selections found in logs.")
            continue

        selected_windows = sorted(selected_coeffs_by_window.keys())
        X_train_flat, X_val_flat, X_test_flat, feat_meta = build_flat_features_for_lag_concat(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            fft_window_sizes=selected_windows,
            selected_coeffs_by_window=selected_coeffs_by_window,
            forecast_lag=int(lag),
            event_index=event_index,
            observation_window_size=observation_window_size,
        )

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state + int(lag),
            n_jobs=-1,
        )
        model.fit(X_train_flat, y_train)

        y_pred_train = model.predict(X_train_flat)
        y_pred_val = model.predict(X_val_flat)
        y_pred_test = model.predict(X_test_flat)

        train_metrics = compute_all_metrics(y_train, y_pred_train, positive_class)
        val_metrics = compute_all_metrics(y_val, y_pred_val, positive_class)
        test_metrics = compute_all_metrics(y_test, y_pred_test, positive_class)
        coeff_tokens = [f"W{m['window_size']}:{m['effective_coeffs']}" for m in feat_meta]

        lag_row = {
            "forecast_lag_min": int(lag),
            "input_start": int(input_start),
            "input_end": int(input_end),
            "num_windows": int(len(coeff_tokens)),
            "selected_coefficients": ",".join(coeff_tokens),
            "total_input_dim": int(X_train_flat.shape[1]),
            "selection_source": "experiment12_results_log",
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
        lag_rows.append(lag_row)
        print(
            f"Lag-level concatenated model (lag={lag}): "
            f"input_dim={lag_row['total_input_dim']}, "
            f"val_css={lag_row['val_css']:.4f}, test_css={lag_row['test_css']:.4f}"
        )

    if not lag_rows:
        raise RuntimeError("No lag-level concatenated result generated from logs.")
    return pd.DataFrame(lag_rows)


def find_elbow_row(curve_df: pd.DataFrame, metric_col: str = "val_css", tolerance: float = 0.005) -> pd.Series:
    ordered = curve_df.sort_values("effective_coeffs")
    best_val = float(ordered[metric_col].max())
    threshold = best_val - tolerance
    plateau = ordered[ordered[metric_col] >= threshold]
    if len(plateau) == 0:
        return ordered.iloc[-1]
    return plateau.iloc[0]


def cross_validate_feature_count(
    X_flat: np.ndarray,
    y: np.ndarray,
    positive_class: int,
    random_state: int = 42,
    cv_splits: int = 3,
) -> dict[str, float]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {
            "cv_folds_used": 0,
            "cv_accuracy_mean": np.nan,
            "cv_f1_mean": np.nan,
            "cv_precision_mean": np.nan,
            "cv_recall_mean": np.nan,
            "cv_tss_mean": np.nan,
            "cv_hss_mean": np.nan,
            "cv_css_mean": np.nan,
            "cv_css_std": np.nan,
        }

    max_splits = int(counts.min())
    folds = min(cv_splits, max_splits)
    if folds < 2:
        return {
            "cv_folds_used": folds,
            "cv_accuracy_mean": np.nan,
            "cv_f1_mean": np.nan,
            "cv_precision_mean": np.nan,
            "cv_recall_mean": np.nan,
            "cv_tss_mean": np.nan,
            "cv_hss_mean": np.nan,
            "cv_css_mean": np.nan,
            "cv_css_std": np.nan,
        }

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    fold_scores: list[dict[str, float]] = []

    for fold_idx, (fit_idx, val_idx) in enumerate(skf.split(X_flat, y)):
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state + fold_idx,
            n_jobs=-1,
        )
        model.fit(X_flat[fit_idx], y[fit_idx])
        y_pred = model.predict(X_flat[val_idx])
        fold_scores.append(compute_all_metrics(y[val_idx], y_pred, positive_class))

    cv_df = pd.DataFrame(fold_scores)
    return {
        "cv_folds_used": folds,
        "cv_accuracy_mean": float(cv_df["accuracy"].mean()),
        "cv_f1_mean": float(cv_df["f1"].mean()),
        "cv_precision_mean": float(cv_df["precision"].mean()),
        "cv_recall_mean": float(cv_df["recall"].mean()),
        "cv_tss_mean": float(cv_df["tss"].mean()),
        "cv_hss_mean": float(cv_df["hss"].mean()),
        "cv_css_mean": float(cv_df["css"].mean()),
        "cv_css_std": float(cv_df["css"].std(ddof=0)),
    }


def build_cv_candidate_k(max_coeffs: int, best_k: int, elbow_k: int) -> list[int]:
    candidates = {1, max_coeffs, best_k, elbow_k}
    for delta in (-2, -1, 1, 2):
        k = elbow_k + delta
        if 1 <= k <= max_coeffs:
            candidates.add(k)
    return sorted(candidates)


def run_experiments(
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_val: list[np.ndarray],
    y_val: np.ndarray,
    X_test: list[np.ndarray],
    y_test: np.ndarray,
    classes: np.ndarray,
    fft_window_sizes: list[int],
    forecast_lags: list[int],
    event_index: int,
    observation_window_size: int,
    random_state: int = 42,
    cv_splits: int = 3,
    elbow_tolerance: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive_class = pick_positive_class(classes)
    print(f"Positive class for scoring: idx={positive_class}, label='{classes[positive_class]}'")

    rows: list[dict[str, float | int | str]] = []
    setup_summaries: list[dict[str, float | int | str]] = []
    cv_rows: list[dict[str, float | int]] = []
    lag_rows: list[dict[str, float | int | str]] = []

    for lag in forecast_lags:
        input_end = event_index - lag
        input_start = input_end - observation_window_size
        print(
            f"\n=== Processing lag={lag} min | Input window=[{input_start}:{input_end}] ==="
        )

        lag_feature_bank: dict[int, dict[str, np.ndarray | int]] = {}
        selected_best_k_by_window: dict[int, int] = {}

        for fft_window_size in fft_window_sizes:
            print(f"\n--- FFT window={fft_window_size} min | lag={lag} min ---")
            X_train_all, max_coeffs, n_slices, channels = build_fft_features_all(
                X_train,
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
            X_val_all, _, _, _ = build_fft_features_all(
                X_val,
                fft_window_size=fft_window_size,
                event_index=event_index,
                observation_window_size=observation_window_size,
                forecast_lag=lag,
            )

            lag_feature_bank[int(fft_window_size)] = {
                "X_train_all": X_train_all,
                "X_val_all": X_val_all,
                "X_test_all": X_test_all,
                "max_coeffs": int(max_coeffs),
                "n_slices": int(n_slices),
            }

            print(
                f"all coefficients available (per slice) for this setup: {max_coeffs} "
                f"[channels={channels}, slices={n_slices}]"
            )
            coeff_options = build_dynamic_coeff_options(max_coeffs)
            print(f"Sweeping coefficient counts: 1..{max_coeffs}")

            setup_rows: list[dict[str, float | int | str]] = []
            progress_stride = max(1, max_coeffs // 10)

            for coeff in coeff_options:
                X_train_spec, k = select_top_k_coeffs(X_train_all, max_coeffs, n_slices, coeff)
                X_test_spec, _ = select_top_k_coeffs(X_test_all, max_coeffs, n_slices, coeff)
                X_val_spec, _ = select_top_k_coeffs(X_val_all, max_coeffs, n_slices, coeff)

                X_train_flat = X_train_spec.reshape(X_train_spec.shape[0], -1)
                X_val_flat = X_val_spec.reshape(X_val_spec.shape[0], -1)
                X_test_flat = X_test_spec.reshape(X_test_spec.shape[0], -1)

                if k == 1 or k == max_coeffs or k % progress_stride == 0:
                    print(
                        f"Running k={k}/{max_coeffs} | "
                        f"rf_input_dim={X_train_flat.shape[1]}"
                    )

                model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_train_flat, y_train)

                y_pred_train = model.predict(X_train_flat)
                y_pred_val = model.predict(X_val_flat)
                y_pred_test = model.predict(X_test_flat)

                train_metrics = compute_all_metrics(y_train, y_pred_train, positive_class)
                val_metrics = compute_all_metrics(y_val, y_pred_val, positive_class)
                test_metrics = compute_all_metrics(y_test, y_pred_test, positive_class)

                row = {
                    "window_size": int(fft_window_size),
                    "forecast_lag_min": int(lag),
                    "input_start": int(input_start),
                    "input_end": int(input_end),
                    "coeff_setting": str(coeff),
                    "max_available_coeffs": int(max_coeffs),
                    "effective_coeffs": int(k),
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
                rows.append(row)
                setup_rows.append(row)

                if k == 1 or k == max_coeffs or k % progress_stride == 0:
                    print(
                        " -> "
                        f"VAL CSS={val_metrics['css']:.4f}, TSS={val_metrics['tss']:.4f}, "
                        f"HSS={val_metrics['hss']:.4f}, F1={val_metrics['f1']:.4f}"
                    )

            setup_df = pd.DataFrame(setup_rows)
            best_row = sort_by_validation_priority(setup_df).iloc[0]
            elbow_row = find_elbow_row(
                setup_df,
                metric_col="val_css",
                tolerance=elbow_tolerance,
            )
            best_k = int(best_row["effective_coeffs"])
            elbow_k = int(elbow_row["effective_coeffs"])
            best_val_css = float(best_row["val_css"])
            selected_best_k_by_window[int(fft_window_size)] = best_k

            cv_candidates = build_cv_candidate_k(
                max_coeffs=max_coeffs,
                best_k=best_k,
                elbow_k=elbow_k,
            )
            setup_cv_rows: list[dict[str, float | int]] = []

            for k in cv_candidates:
                X_train_spec, _ = select_top_k_coeffs(X_train_all, max_coeffs, n_slices, k)
                X_train_flat = X_train_spec.reshape(X_train_spec.shape[0], -1)
                cv_metrics = cross_validate_feature_count(
                    X_train_flat,
                    y_train,
                    positive_class=positive_class,
                    random_state=random_state + int(fft_window_size) + int(lag) + int(k),
                    cv_splits=cv_splits,
                )
                cv_row = {
                    "window_size": int(fft_window_size),
                    "forecast_lag_min": int(lag),
                    "effective_coeffs": int(k),
                    "max_available_coeffs": int(max_coeffs),
                    **cv_metrics,
                }
                cv_rows.append(cv_row)
                setup_cv_rows.append(cv_row)

            cv_selected_k = elbow_k
            cv_selected_css = np.nan
            cv_selected_tss = np.nan
            cv_selected_hss = np.nan
            cv_selected_f1 = np.nan
            cv_folds_used = np.nan

            if setup_cv_rows:
                cv_df = pd.DataFrame(setup_cv_rows)
                if cv_df["cv_css_mean"].notna().any():
                    cv_best = cv_df.sort_values(
                        by=[
                            "cv_css_mean",
                            "cv_tss_mean",
                            "cv_hss_mean",
                            "cv_f1_mean",
                            "effective_coeffs",
                        ],
                        ascending=[False, False, False, False, True],
                    ).iloc[0]
                    cv_selected_k = int(cv_best["effective_coeffs"])
                    cv_selected_css = float(cv_best["cv_css_mean"])
                    cv_selected_tss = float(cv_best["cv_tss_mean"])
                    cv_selected_hss = float(cv_best["cv_hss_mean"])
                    cv_selected_f1 = float(cv_best["cv_f1_mean"])
                    cv_folds_used = float(cv_best["cv_folds_used"])

            setup_summaries.append(
                {
                    "window_size": int(fft_window_size),
                    "forecast_lag_min": int(lag),
                    "input_start": int(input_start),
                    "input_end": int(input_end),
                    "max_available_coeffs": int(max_coeffs),
                    "best_val_k": best_k,
                    "best_val_css": best_val_css,
                    "best_val_tss": float(best_row["val_tss"]),
                    "best_val_hss": float(best_row["val_hss"]),
                    "best_val_f1": float(best_row["val_f1"]),
                    "elbow_k": elbow_k,
                    "elbow_val_css": float(elbow_row["val_css"]),
                    "elbow_tolerance": float(elbow_tolerance),
                    "elbow_threshold_css": float(best_val_css - elbow_tolerance),
                    "cv_selected_k": int(cv_selected_k),
                    "cv_css_mean": cv_selected_css,
                    "cv_tss_mean": cv_selected_tss,
                    "cv_hss_mean": cv_selected_hss,
                    "cv_f1_mean": cv_selected_f1,
                    "cv_folds_used": cv_folds_used,
                    "cv_candidates": ",".join(str(k) for k in cv_candidates),
                }
            )
            print(
                f"Selected k for W{fft_window_size}_lag{lag}: "
                f"best_val_k={best_k}, elbow_k={elbow_k}, cv_selected_k={cv_selected_k}"
            )

        concat_train_parts: list[np.ndarray] = []
        concat_val_parts: list[np.ndarray] = []
        concat_test_parts: list[np.ndarray] = []
        concat_window_tokens: list[str] = []

        for fft_window_size in fft_window_sizes:
            window = int(fft_window_size)
            if window not in lag_feature_bank or window not in selected_best_k_by_window:
                continue

            bank = lag_feature_bank[window]
            max_coeffs = int(bank["max_coeffs"])
            n_slices = int(bank["n_slices"])
            k = int(selected_best_k_by_window[window])

            X_train_spec, k_eff = select_top_k_coeffs(
                bank["X_train_all"],
                max_coeffs,
                n_slices,
                k,
            )
            X_val_spec, _ = select_top_k_coeffs(
                bank["X_val_all"],
                max_coeffs,
                n_slices,
                k,
            )
            X_test_spec, _ = select_top_k_coeffs(
                bank["X_test_all"],
                max_coeffs,
                n_slices,
                k,
            )

            concat_train_parts.append(X_train_spec.reshape(X_train_spec.shape[0], -1))
            concat_val_parts.append(X_val_spec.reshape(X_val_spec.shape[0], -1))
            concat_test_parts.append(X_test_spec.reshape(X_test_spec.shape[0], -1))
            concat_window_tokens.append(f"W{window}:{k_eff}")

        if not concat_train_parts:
            print(f"Skipping lag={lag}: no selected window features.")
            continue

        X_train_concat = np.concatenate(concat_train_parts, axis=1)
        X_val_concat = np.concatenate(concat_val_parts, axis=1)
        X_test_concat = np.concatenate(concat_test_parts, axis=1)

        lag_model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state + int(lag),
            n_jobs=-1,
        )
        lag_model.fit(X_train_concat, y_train)

        y_pred_train = lag_model.predict(X_train_concat)
        y_pred_val = lag_model.predict(X_val_concat)
        y_pred_test = lag_model.predict(X_test_concat)

        train_metrics = compute_all_metrics(y_train, y_pred_train, positive_class)
        val_metrics = compute_all_metrics(y_val, y_pred_val, positive_class)
        test_metrics = compute_all_metrics(y_test, y_pred_test, positive_class)

        lag_row = {
            "forecast_lag_min": int(lag),
            "input_start": int(input_start),
            "input_end": int(input_end),
            "num_windows": int(len(concat_window_tokens)),
            "selected_coefficients": ",".join(concat_window_tokens),
            "total_input_dim": int(X_train_concat.shape[1]),
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
        lag_rows.append(lag_row)
        print(
            f"Lag-level concatenated model (lag={lag}): "
            f"input_dim={lag_row['total_input_dim']}, val_css={lag_row['val_css']:.4f}, "
            f"test_css={lag_row['test_css']:.4f}"
        )

    if not rows:
        raise RuntimeError("No experiment result generated.")
    if not lag_rows:
        raise RuntimeError("No lag-level concatenated result generated.")

    return (
        pd.DataFrame(rows),
        pd.DataFrame(setup_summaries),
        pd.DataFrame(cv_rows),
        pd.DataFrame(lag_rows),
    )


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

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.get_splits()

    event_onset_index = 720
    observation_window_size = 360
    fft_window_sizes = [360, 180, 90, 45]
    forecast_lags = [5, 15, 30, 60, 120]

    experiment12_report_dir = data_root / "reports" / "experiment12"
    results_log_path = find_latest_results_log(experiment12_report_dir, pattern="experiment12_results_*.csv")
    print(f"Using Experiment 12 log: {results_log_path}")

    results_df = load_results_log(results_log_path)
    results_df = results_df[
        results_df["window_size"].astype(int).isin([int(w) for w in fft_window_sizes])
        & results_df["forecast_lag_min"].astype(int).isin([int(l) for l in forecast_lags])
    ].copy()
    if len(results_df) == 0:
        raise RuntimeError(
            "No rows from Experiment 12 logs match requested window sizes and forecast lags."
        )

    setup_summary_df = build_setup_summary_from_logs(
        results_df=results_df,
        elbow_tolerance=0.005,
    )
    lag_results_df = run_lag_concatenated_from_logs(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=dataset.classes_,
        fft_window_sizes=fft_window_sizes,
        forecast_lags=forecast_lags,
        setup_summary_df=setup_summary_df,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        random_state=42,
    )

    output_dir = data_root / "reports" / "experiment14"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"experiment14_results_{stamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    setup_summary_path = output_dir / f"experiment14_feature_selection_summary_{stamp}.csv"
    setup_summary_df.to_csv(setup_summary_path, index=False)
    print(f"Saved feature-selection summary: {setup_summary_path}")

    lag_results_path = output_dir / f"experiment14_lag_concatenated_results_{stamp}.csv"
    lag_results_df.to_csv(lag_results_path, index=False)
    print(f"Saved lag-level concatenated results: {lag_results_path}")

    lag_level_val_path = output_dir / f"experiment14_lag_level_validation_by_lag_{stamp}.png"
    plot_lag_level_validation_by_lag(lag_results_df, lag_level_val_path)
    print(f"Saved lag-level validation-by-lag plot: {lag_level_val_path}")

    best = select_best_lag_result(lag_results_df)
    best_path = output_dir / f"experiment14_best_lag_config_{stamp}.csv"
    best.to_frame().T.to_csv(best_path, index=False)

    models_dir = output_dir / "saved_models"
    counterfactuals_dir = output_dir / "counterfactuals"
    sep_cfe_lib_dir = Path("/Users/pranjal/PycharmProjects/SolarEnergyParticlePrediction/src/SEP_CFE_DiCE")
    sep_project_root = sep_cfe_lib_dir.parents[1]

    calibration_all_df, feature_importance_df, counterfactual_summary_df = run_calibration_and_feature_importance_for_all_lags(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=dataset.classes_,
        fft_window_sizes=fft_window_sizes,
        forecast_lags=forecast_lags,
        setup_summary_df=setup_summary_df,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        output_dir=output_dir,
        stamp=stamp,
        random_state=42,
        top_n_features=25,
        channel_names=target_channels,
        save_models_dir=models_dir,
        enable_constrained_dice=True,
        sep_project_root=sep_project_root,
        counterfactuals_output_dir=counterfactuals_dir,
        counterfactuals_per_query=3,
        max_queries_per_lag=5,
        outcome_name="Event_Y_N",
    )

    calibration_all_path = output_dir / f"experiment14_all_lags_calibration_summary_{stamp}.csv"
    calibration_all_df.to_csv(calibration_all_path, index=False)
    print(f"Saved all-lags calibration summary: {calibration_all_path}")

    feature_importance_path = output_dir / f"experiment14_all_lags_feature_importance_{stamp}.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)
    print(f"Saved all-lags feature importance table: {feature_importance_path}")

    counterfactual_summary_path = output_dir / f"experiment14_all_lags_counterfactual_summary_{stamp}.csv"
    counterfactual_summary_df.to_csv(counterfactual_summary_path, index=False)
    print(f"Saved all-lags counterfactual summary: {counterfactual_summary_path}")
    print(f"Saved lag models under: {models_dir}")
    print(f"Saved lag counterfactual tables under: {counterfactuals_dir}")

    print("\n" + "=" * 70)
    print(
        "Best lag-level configuration (sorted by val_css, val_tss, val_hss, val_f1 "
        "with test metrics as final tie-breakers):"
    )
    for col in [
        "forecast_lag_min",
        "num_windows",
        "selected_coefficients",
        "total_input_dim",
        "val_css",
        "val_tss",
        "val_hss",
        "val_f1",
        "test_css",
        "test_tss",
        "test_hss",
        "test_f1",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "test_accuracy",
        "test_precision",
        "test_recall",
    ]:
        print(f"{col}: {best[col]}")
    print("=" * 70)
    print(f"Saved best lag config: {best_path}")

    if len(setup_summary_df) > 0:
        print("\nPer-setup selection snapshot (derived from Experiment 12 logs):")
        summary_cols = [
            "window_size",
            "forecast_lag_min",
            "max_available_coeffs",
            "best_val_k",
            "elbow_k",
            "best_val_css",
            "elbow_val_css",
            "selection_source",
        ]
        print(setup_summary_df[summary_cols].to_string(index=False))
  


if __name__ == "__main__":
    main()
