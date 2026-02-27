"""Experiment 12: Run RandomForest on FFT magnitude+phase features using an
event-anchored window. For each forecast lag, data is extracted as
[event_index - lag - observation_window : event_index - lag], then
split into non-overlapping FFT windows across multiple sizes.

Instead of pre-defining a small set of coefficient counts, this script builds
data-driven validation curves by sweeping k=1..max_coeffs (per setup), where
"k" is the number of magnitude and phase coefficients retained per channel per
slice.

For each window/lag setup it:
  1) Fits models across increasing FFT coefficient counts.
  2) Plots performance versus number of coefficients.
  3) Chooses an elbow/plateau point as the smallest k within a tolerance of
     the best validation CSS.
  4) Runs cross-validation on candidate k values near elbow/best to support
     automated feature-count selection.
"""

from __future__ import annotations

import os
import math
from datetime import datetime
from pathlib import Path

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
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

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
        ax.set_title(f"{split_name} | Brier raw={brier_raw:.4f}, cal={brier_cal:.4f}")
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positive_class = pick_positive_class(classes)
    print(f"Positive class for scoring: idx={positive_class}, label='{classes[positive_class]}'")

    rows: list[dict[str, float | int | str]] = []
    setup_summaries: list[dict[str, float | int | str]] = []
    cv_rows: list[dict[str, float | int]] = []

    for fft_window_size in fft_window_sizes:
        for lag in forecast_lags:
            input_end = event_index - lag
            input_start = input_end - observation_window_size
            print(
                f"\n=== FFT window={fft_window_size} min | lag={lag} min "
                f"| Input window=[{input_start}:{input_end}] ==="
            )

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

    if not rows:
        raise RuntimeError("No experiment result generated.")

    return pd.DataFrame(rows), pd.DataFrame(setup_summaries), pd.DataFrame(cv_rows)


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

    results_df, setup_summary_df, cv_results_df = run_experiments(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=dataset.classes_,
        fft_window_sizes=fft_window_sizes,
        forecast_lags=forecast_lags,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        random_state=42,
        cv_splits=3,
        elbow_tolerance=0.005,
    )

    output_dir = data_root / "reports" / "experiment12"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"experiment12_results_{stamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    setup_summary_path = output_dir / f"experiment12_feature_selection_summary_{stamp}.csv"
    setup_summary_df.to_csv(setup_summary_path, index=False)
    print(f"Saved feature-selection summary: {setup_summary_path}")

    cv_path = output_dir / f"experiment12_cv_feature_selection_{stamp}.csv"
    cv_results_df.to_csv(cv_path, index=False)
    print(f"Saved CV feature-selection table: {cv_path}")

    val_lineplot_path = output_dir / f"experiment12_val_metrics_lineplot_{stamp}.png"
    plot_metric_lineplots(results_df, val_lineplot_path, split_prefix="val")
    print(f"Saved validation line plot: {val_lineplot_path}")

    val_consolidated_path = output_dir / f"experiment12_val_consolidated_metrics_{stamp}.png"
    plot_consolidated_metrics_by_setup(results_df, val_consolidated_path, split_prefix="val")
    print(f"Saved validation consolidated metrics plot: {val_consolidated_path}")

    elbow_curve_path = output_dir / f"experiment12_validation_elbow_curves_{stamp}.png"
    plot_validation_elbow_curves(results_df, setup_summary_df, elbow_curve_path)
    print(f"Saved validation/elbow curves: {elbow_curve_path}")

    test_lineplot_path = output_dir / f"experiment12_test_metrics_lineplot_{stamp}.png"
    plot_metric_lineplots(results_df, test_lineplot_path, split_prefix="test")
    print(f"Saved test line plot: {test_lineplot_path}")

    test_consolidated_path = output_dir / f"experiment12_test_consolidated_metrics_{stamp}.png"
    plot_consolidated_metrics_by_setup(results_df, test_consolidated_path, split_prefix="test")
    print(f"Saved test consolidated metrics plot: {test_consolidated_path}")

    best = select_best(results_df)
    best_path = output_dir / f"experiment12_best_config_{stamp}.csv"
    best.to_frame().T.to_csv(best_path, index=False)

    positive_class = pick_positive_class(dataset.classes_)
    best_window = int(best["window_size"])
    best_lag = int(best["forecast_lag_min"])
    best_k = int(best["effective_coeffs"])
    X_train_flat, X_val_flat, X_test_flat, feat_meta = build_flat_features_for_config(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        fft_window_size=best_window,
        forecast_lag=best_lag,
        coeff_count=best_k,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
    )

    base_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    base_model.fit(X_train_flat, y_train)

    val_prob_raw = get_positive_class_probability(base_model, X_val_flat, positive_class)
    test_prob_raw = get_positive_class_probability(base_model, X_test_flat, positive_class)

    calibrator = None
    calibration_status = "calibrated_sigmoid_prefit"
    try:
        calibrator = calibrate_prefit_sigmoid(base_model, X_val_flat, y_val)
        if calibrator is None:
            calibration_status = "skipped_single_class_validation"
    except Exception as exc:
        calibration_status = f"calibration_failed: {exc}"
        print(f"Calibration warning: {exc}")

    if calibrator is not None:
        val_prob_cal = get_positive_class_probability(calibrator, X_val_flat, positive_class)
        test_prob_cal = get_positive_class_probability(calibrator, X_test_flat, positive_class)
    else:
        val_prob_cal = val_prob_raw.copy()
        test_prob_cal = test_prob_raw.copy()

    y_val_bin = (y_val == positive_class).astype(int)
    y_test_bin = (y_test == positive_class).astype(int)
    calibration_plot_path = output_dir / f"experiment12_best_config_calibration_{stamp}.png"
    plot_calibration_curves(
        y_val_bin=y_val_bin,
        y_test_bin=y_test_bin,
        val_prob_raw=val_prob_raw,
        test_prob_raw=test_prob_raw,
        val_prob_cal=val_prob_cal,
        test_prob_cal=test_prob_cal,
        output_path=calibration_plot_path,
    )
    print(f"Saved calibration plot: {calibration_plot_path}")

    calibration_summary_path = output_dir / f"experiment12_best_config_calibration_{stamp}.csv"
    calibration_summary = pd.DataFrame(
        [
            {
                "window_size": best_window,
                "forecast_lag_min": best_lag,
                "effective_coeffs": int(feat_meta["effective_coeffs"]),
                "max_available_coeffs": int(feat_meta["max_available_coeffs"]),
                "calibration_status": calibration_status,
                "val_brier_raw": float(brier_score_loss(y_val_bin, val_prob_raw)),
                "val_brier_calibrated": float(brier_score_loss(y_val_bin, val_prob_cal)),
                "test_brier_raw": float(brier_score_loss(y_test_bin, test_prob_raw)),
                "test_brier_calibrated": float(brier_score_loss(y_test_bin, test_prob_cal)),
            }
        ]
    )
    calibration_summary.to_csv(calibration_summary_path, index=False)
    print(f"Saved calibration summary: {calibration_summary_path}")

    print("\n" + "=" * 70)
    print(
        "Best configuration (sorted by val_css, val_tss, val_hss, val_f1 "
        "with test metrics as final tie-breakers):"
    )
    for col in [
        "window_size",
        "forecast_lag_min",
        "coeff_setting",
        "max_available_coeffs",
        "effective_coeffs",
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
    print(f"Saved best config: {best_path}")

    if len(setup_summary_df) > 0:
        print("\nPer-setup selection snapshot (best/plateau/CV):")
        summary_cols = [
            "window_size",
            "forecast_lag_min",
            "max_available_coeffs",
            "best_val_k",
            "elbow_k",
            "cv_selected_k",
            "best_val_css",
            "elbow_val_css",
            "cv_css_mean",
        ]
        print(setup_summary_df[summary_cols].to_string(index=False))
  


if __name__ == "__main__":
    main()
