"""Experiment 14: Lag-wise models with window-wise top-percentage concatenation.

For each forecast lag:
1) Train a baseline RandomForest for each FFT window size using all FFT features.
2) Within each (lag, window), rank features by baseline importance.
3) For each top-percent in {5, 10, 15, 20, 25}:
   - keep top p% features from each window for that lag,
   - concatenate selected window features in window order,
   - train one RandomForest model for that lag/top% pair,
   - evaluate and save outputs.

Outputs are intentionally overwritten on each run.
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.fft import rfft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
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
    def _parse_event_time(filename: str) -> pd.Timestamp:
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


def count_features_for_cumulative_importance(
    feature_importances: np.ndarray,
    threshold: float,
    strict: bool = False,
) -> tuple[int, float]:
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}.")

    importances = np.asarray(feature_importances, dtype=np.float64).ravel()
    importances = importances[np.isfinite(importances)]
    if importances.size == 0:
        return 0, 0.0

    total_importance = float(importances.sum())
    if total_importance <= 0.0:
        return 0, 0.0

    sorted_importances = np.sort(importances)[::-1]
    cumulative = np.cumsum(sorted_importances) / total_importance

    side = "right" if strict else "left"
    idx = int(np.searchsorted(cumulative, threshold, side=side))
    idx = min(idx, sorted_importances.size - 1)
    return int(idx + 1), float(cumulative[idx])


def select_top_percentage_indices(feature_importances: np.ndarray, top_fraction: float) -> tuple[np.ndarray, int]:
    frac = float(top_fraction)
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"top_fraction must be in (0, 1], got {frac}.")

    importances = np.asarray(feature_importances, dtype=np.float64).ravel()
    n_features = int(importances.size)
    if n_features == 0:
        return np.array([], dtype=np.int64), 0

    selected_count = max(1, int(math.ceil(frac * n_features)))
    ranked_idx = np.argsort(importances)[::-1]
    return ranked_idx[:selected_count].astype(np.int64), int(selected_count)


def build_fft_feature_names(
    window_size: int,
    lag: int,
    channels: int,
    channel_names: list[str],
    n_slices: int,
    max_coeffs: int,
) -> list[str]:
    if len(channel_names) != channels:
        channel_labels = [f"ch{i}" for i in range(channels)]
    else:
        channel_labels = channel_names

    names: list[str] = []
    for c_fft in range(2 * channels):
        is_mag = c_fft < channels
        kind = "mag" if is_mag else "phase"
        c = c_fft if is_mag else (c_fft - channels)

        for s in range(n_slices):
            for coeff in range(max_coeffs):
                names.append(
                    f"w{int(window_size)}_lag{int(lag)}_{kind}_{channel_labels[c]}_slice{s}_coeff{coeff}"
                )
    return names


def percent_key(top_fraction: float) -> int:
    return int(round(float(top_fraction) * 100.0))


def plot_lag_toppercent_training_summary(
    results_df: pd.DataFrame,
    event_index: int,
    observation_window_size: int,
    output_path: Path,
):
    if len(results_df) == 0:
        return

    df = results_df.copy().sort_values(["forecast_lag_min", "top_percent"])
    top_percents = sorted(df["top_percent"].astype(float).unique().tolist())
    lags = sorted(df["forecast_lag_min"].astype(int).unique().tolist())
    x = np.arange(len(top_percents), dtype=np.float64)
    x_labels = [f"{int(round(p * 100.0))}%" for p in top_percents]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    ax_val, ax_test, ax_feat = axes

    for lag in lags:
        lag_df = df[df["forecast_lag_min"].astype(int) == int(lag)].sort_values("top_percent")
        if len(lag_df) == 0:
            continue
        input_end = int(event_index - lag)
        input_start = int(input_end - observation_window_size)
        lag_label = f"lag {lag} [{input_start}:{input_end}]"

        ax_val.plot(x, lag_df["val_css"].astype(float).values, marker="o", linewidth=1.5, label=lag_label)
        ax_test.plot(x, lag_df["test_css"].astype(float).values, marker="o", linewidth=1.5, label=f"lag {lag}")
        ax_feat.plot(
            x,
            lag_df["total_concat_features"].astype(float).values,
            marker="o",
            linewidth=1.5,
            label=f"lag {lag}",
        )

    ax_val.set_title("Validation CSS by top%")
    ax_val.set_ylabel("val_css")
    ax_val.set_ylim(0.0, 1.0)
    ax_val.grid(True, alpha=0.25)
    ax_val.legend(frameon=False, fontsize=8, loc="lower right")

    ax_test.set_title("Test CSS by top%")
    ax_test.set_ylabel("test_css")
    ax_test.set_ylim(0.0, 1.0)
    ax_test.grid(True, alpha=0.25)

    ax_feat.set_title("Concatenated feature count by top%")
    ax_feat.set_ylabel("total_concat_features")
    ax_feat.grid(True, alpha=0.25)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Top-% selected per window")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_feature_importance_line_grid(
    model_artifacts: list[dict],
    results_df: pd.DataFrame,
    forecast_lags: list[int],
    top_percents: list[float],
    output_path: Path,
):
    if not model_artifacts:
        return

    lag_order = [int(l) for l in forecast_lags]
    top_keys = [int(percent_key(p)) for p in top_percents]
    if not lag_order or not top_keys:
        return

    artifact_map: dict[tuple[int, int], dict] = {}
    for artifact in model_artifacts:
        key = (int(artifact["forecast_lag_min"]), int(artifact["top_percent_key"]))
        artifact_map[key] = artifact

    summary_map: dict[tuple[int, int], tuple[float, float]] = {}
    if len(results_df) > 0:
        df = results_df.copy()
        df["top_percent_key"] = (df["top_percent"].astype(float) * 100.0).round().astype(int)
        for _, row in df.iterrows():
            key = (int(row["forecast_lag_min"]), int(row["top_percent_key"]))
            summary_map[key] = (float(row["val_css"]), float(row["test_css"]))

    nrows, ncols = len(lag_order), len(top_keys)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.6 * nrows), sharey=True)
    axes = np.asarray(axes).reshape(nrows, ncols)

    window_colors = ["#dbe9ff", "#ffe8d6", "#e2f3e2", "#f0e0ff"]
    if artifact_map:
        first_artifact = next(iter(artifact_map.values()))
        example_windows = [int(w) for w in first_artifact.get("windows", [])]
    else:
        example_windows = []

    for r, lag in enumerate(lag_order):
        for c, top_key in enumerate(top_keys):
            ax = axes[r, c]
            artifact = artifact_map.get((int(lag), int(top_key)))
            if artifact is None:
                ax.axis("off")
                continue

            importances = np.asarray(artifact["model"].feature_importances_, dtype=np.float64).ravel()
            x = np.arange(importances.size, dtype=np.int64)

            windows = [int(w) for w in artifact.get("windows", sorted(artifact["selected_feature_indices_by_window"].keys()))]
            selected_idx_by_window = artifact["selected_feature_indices_by_window"]

            start = 0
            for i, window in enumerate(windows):
                count = int(len(selected_idx_by_window.get(int(window), [])))
                if count <= 0:
                    continue
                end = start + count
                ax.axvspan(start, max(start, end - 1), color=window_colors[i % len(window_colors)], alpha=0.35, linewidth=0)
                if start > 0:
                    ax.axvline(start, color="#6f6f6f", linewidth=0.6, alpha=0.9)
                start = end

            ax.plot(x, importances, color="#1f77b4", linewidth=0.9)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, max(1, int(importances.size - 1)))
            ax.grid(True, axis="y", alpha=0.2)
            ax.tick_params(labelsize=7)

            if r == 0:
                ax.set_title(f"top {top_key}%", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"lag {lag}\nimportance", fontsize=8)
            if r == nrows - 1:
                ax.set_xlabel("feature idx", fontsize=8)

            val_test = summary_map.get((int(lag), int(top_key)))
            if val_test is not None:
                val_css, test_css = val_test
                ax.text(
                    0.02,
                    0.98,
                    f"V:{val_css:.3f}  T:{test_css:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
                )

    legend_handles = [Line2D([0], [0], color="#1f77b4", lw=1.2, label="feature importance")]
    for i, window in enumerate(example_windows):
        legend_handles.append(
            Patch(facecolor=window_colors[i % len(window_colors)], edgecolor="none", alpha=0.35, label=f"window {window}")
        )

    fig.suptitle("Experiment 14: Feature importance lines (5x5) with window-segment shading", fontsize=12)
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=max(2, len(legend_handles)), frameon=False, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_features_needed_for_half_importance(results_df: pd.DataFrame, output_path: Path):
    if len(results_df) == 0:
        return

    plot_df = results_df.copy()
    top_percents = sorted(plot_df["top_percent"].astype(float).unique().tolist())
    lags = sorted(plot_df["forecast_lag_min"].astype(int).unique().tolist())
    if not top_percents or not lags:
        return

    x = np.arange(len(top_percents), dtype=np.float64)
    width = 0.8 / float(len(lags))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, lag in enumerate(lags):
        lag_df = plot_df[plot_df["forecast_lag_min"].astype(int) == int(lag)].copy()
        lag_df = lag_df.sort_values("top_percent")
        y = lag_df["feature_count_cumimp_gt50"].astype(float).values
        offsets = x + (i - (len(lags) - 1) / 2.0) * width
        bars = ax.bar(offsets, y, width=width, label=f"lag {lag}")
        for bar in bars:
            val = float(bar.get_height())
            ax.text(
                float(bar.get_x() + bar.get_width() / 2.0),
                val,
                f"{int(round(val))}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(round(p * 100.0))}%" for p in top_percents])
    ax.set_title("Experiment 14: #features needed to exceed 50% cumulative importance")
    ax.set_xlabel("Top-% selected per window")
    ax.set_ylabel("Feature count to exceed 50% cumulative importance")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(len(lags), 5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def run_lag_window_concatenated_top_percent_experiment(
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

        window_cache: dict[int, dict] = {}

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

            baseline_model = RandomForestClassifier(
                n_estimators=300,
                random_state=random_state + int(fft_window_size) * 100 + int(lag),
                n_jobs=-1,
            )
            baseline_model.fit(X_train_flat, y_train)
            baseline_importances = np.asarray(baseline_model.feature_importances_, dtype=np.float64)

            total_features = int(baseline_importances.size)
            lag_half_count, lag_half_reached = count_features_for_cumulative_importance(
                baseline_importances,
                threshold=0.50,
                strict=True,
            )

            feature_names = build_fft_feature_names(
                window_size=fft_window_size,
                lag=lag,
                channels=channels,
                channel_names=channel_names,
                n_slices=n_slices,
                max_coeffs=max_coeffs,
            )

            selected_by_percent: dict[int, np.ndarray] = {}
            for frac in top_percents:
                pkey = percent_key(frac)
                selected_idx, selected_count = select_top_percentage_indices(baseline_importances, frac)
                selected_by_percent[pkey] = selected_idx

                selection_rows.append(
                    {
                        "forecast_lag_min": int(lag),
                        "window_size": int(fft_window_size),
                        "top_percent": float(frac),
                        "top_percent_label": f"{pkey}%",
                        "total_features_in_window_model": total_features,
                        "selected_feature_count": int(selected_count),
                        "selected_feature_fraction": float(selected_count) / float(total_features),
                        "window_model_feature_count_cumimp_gt50": int(lag_half_count),
                        "window_model_cumimp_reached_gt50": float(lag_half_reached),
                        "max_available_coeffs": int(max_coeffs),
                        "n_slices": int(n_slices),
                        "channels": int(channels),
                    }
                )

            window_cache[int(fft_window_size)] = {
                "X_train": X_train_flat,
                "X_val": X_val_flat,
                "X_test": X_test_flat,
                "feature_names": feature_names,
                "selected_by_percent": selected_by_percent,
                "total_features": int(total_features),
            }

        for frac in top_percents:
            pkey = percent_key(frac)
            X_train_parts: list[np.ndarray] = []
            X_val_parts: list[np.ndarray] = []
            X_test_parts: list[np.ndarray] = []
            concat_feature_names: list[str] = []
            selected_idx_by_window: dict[int, list[int]] = {}

            for fft_window_size in fft_window_sizes:
                cache = window_cache[int(fft_window_size)]
                selected_idx = cache["selected_by_percent"][pkey]

                X_train_parts.append(cache["X_train"][:, selected_idx])
                X_val_parts.append(cache["X_val"][:, selected_idx])
                X_test_parts.append(cache["X_test"][:, selected_idx])

                selected_idx_list = [int(i) for i in selected_idx.tolist()]
                selected_idx_by_window[int(fft_window_size)] = selected_idx_list
                concat_feature_names.extend([cache["feature_names"][i] for i in selected_idx_list])

            X_train_concat = np.concatenate(X_train_parts, axis=1)
            X_val_concat = np.concatenate(X_val_parts, axis=1)
            X_test_concat = np.concatenate(X_test_parts, axis=1)

            model = RandomForestClassifier(
                n_estimators=300,
                random_state=random_state + int(lag) * 1000 + int(pkey),
                n_jobs=-1,
            )
            model.fit(X_train_concat, y_train)

            y_pred_train = model.predict(X_train_concat)
            y_pred_val = model.predict(X_val_concat)
            y_pred_test = model.predict(X_test_concat)

            train_metrics = compute_all_metrics(y_train, y_pred_train, positive_class)
            val_metrics = compute_all_metrics(y_val, y_pred_val, positive_class)
            test_metrics = compute_all_metrics(y_test, y_pred_test, positive_class)

            model_importances = np.asarray(model.feature_importances_, dtype=np.float64)
            model_half_count, model_half_reached = count_features_for_cumulative_importance(
                model_importances,
                threshold=0.50,
                strict=True,
            )

            result_row: dict[str, float | int | str] = {
                "forecast_lag_min": int(lag),
                "top_percent": float(frac),
                "top_percent_label": f"{pkey}%",
                "windows_used": ",".join(str(w) for w in fft_window_sizes),
                "total_concat_features": int(X_train_concat.shape[1]),
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

            for fft_window_size in fft_window_sizes:
                win_total = int(window_cache[int(fft_window_size)]["total_features"])
                win_selected = int(len(selected_idx_by_window[int(fft_window_size)]))
                result_row[f"window{int(fft_window_size)}_total_features"] = win_total
                result_row[f"window{int(fft_window_size)}_selected_features"] = win_selected

            results_rows.append(result_row)

            model_artifacts.append(
                {
                    "forecast_lag_min": int(lag),
                    "top_percent": float(frac),
                    "top_percent_key": int(pkey),
                    "windows": [int(w) for w in fft_window_sizes],
                    "model": model,
                    "feature_names": concat_feature_names,
                    "selected_feature_indices_by_window": selected_idx_by_window,
                    "total_concat_features": int(X_train_concat.shape[1]),
                    "feature_count_cumimp_gt50": int(model_half_count),
                    "cumimp_reached_at_gt50": float(model_half_reached),
                }
            )

            print(
                f"Trained model lag{lag}_top{pkey}% | "
                f"concat_features={X_train_concat.shape[1]} | "
                f"val_css={val_metrics['css']:.4f} | test_css={test_metrics['css']:.4f}"
            )

    if not results_rows:
        raise RuntimeError("No model results generated.")

    return pd.DataFrame(results_rows), pd.DataFrame(selection_rows), model_artifacts


def save_model_artifacts(model_artifacts: list[dict], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for artifact in model_artifacts:
        lag = int(artifact["forecast_lag_min"])
        top_key = int(artifact["top_percent_key"])
        path = output_dir / f"experiment14_rf_lag{lag}_top{top_key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        saved_paths.append(path)

    return saved_paths


def remove_matching_files(directory: Path, pattern: str):
    if not directory.exists():
        return
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()


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
    fft_window_sizes = [45, 90, 180, 360]
    forecast_lags = [5, 15, 30, 60, 120]
    top_percents = [0.05, 0.10, 0.15, 0.20, 0.25]

    results_df, selection_df, model_artifacts = run_lag_window_concatenated_top_percent_experiment(
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

    output_dir = data_root / "reports" / "experiment14"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "experiment14_lag_models_concatenated_windows_results.csv"
    selection_path = output_dir / "experiment14_window_selection_details.csv"
    results_df.to_csv(results_path, index=False)
    selection_df.to_csv(selection_path, index=False)

    models_dir = output_dir / "saved_models"
    remove_matching_files(models_dir, "experiment14_rf_*.pkl")
    saved_model_paths = save_model_artifacts(model_artifacts, models_dir)

    summary_plot_path = output_dir / "experiment14_lag_toppercent_training_summary.png"
    plot_lag_toppercent_training_summary(
        results_df=results_df,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        output_path=summary_plot_path,
    )

    feature_importance_grid_path = output_dir / "experiment14_feature_importance_line_grid_5x5.png"
    plot_feature_importance_line_grid(
        model_artifacts=model_artifacts,
        results_df=results_df,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        output_path=feature_importance_grid_path,
    )

    half_importance_plot_path = output_dir / "experiment14_features_required_cumimp_gt50_by_lag_toppercent.png"
    plot_features_needed_for_half_importance(results_df, half_importance_plot_path)

    print("\n" + "=" * 80)
    print("Experiment 14 (lag-wise models from per-window top-percent concat) finished.")
    print(f"Saved results CSV: {results_path}")
    print(f"Saved lag-selection CSV: {selection_path}")
    print(f"Saved lag/top% training summary plot: {summary_plot_path}")
    print(f"Saved 5x5 feature-importance line grid: {feature_importance_grid_path}")
    print(f"Saved >50% cumulative-importance count plot: {half_importance_plot_path}")
    print(f"Saved models: {len(saved_model_paths)} files in {models_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
