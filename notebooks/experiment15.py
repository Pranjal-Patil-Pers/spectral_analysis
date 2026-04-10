"""Experiment 15: Lag-wise models with global top-percentage selection.

For each forecast lag:
1) Build FFT features for each window size.
2) Concatenate all window features into one lag-level matrix.
3) Train one baseline RandomForest per lag on the full concatenated matrix.
4) For each top-percent in {5, 10, 15, 25, 50, 100}:
   - keep top p% features from the concatenated lag-level feature space,
   - reorder selected features by window for plotting compatibility,
   - train one RandomForest model for that lag/top% pair,
   - evaluate and save outputs.

Outputs are timestamped on each run.
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


def channel_short_name(channel_name: str) -> str:
    low = str(channel_name).strip().lower()
    if low.startswith("p3"):
        return "p3"
    if low.startswith("p5"):
        return "p5"
    if low.startswith("p7"):
        return "p7"
    if low in {"long", "long_xray", "long_xray_channel"}:
        return "long_xray"

    cleaned = "".join(ch if ch.isalnum() else "_" for ch in low).strip("_")
    return cleaned if cleaned else "unknown"


def window_size_from_feature_name(feature_name: str) -> int | None:
    name = str(feature_name)
    if not name.startswith("w"):
        return None
    prefix = name.split("_", 1)[0]
    num = prefix[1:]
    if not num.isdigit():
        return None
    return int(num)


def channel_name_from_feature_name(feature_name: str, channel_names: list[str]) -> str | None:
    name = str(feature_name)
    for channel in channel_names:
        if f"_{channel}_slice" in name:
            return str(channel)
    return None


def slice_index_from_feature_name(feature_name: str) -> int | None:
    name = str(feature_name)
    marker = "_slice"
    pos = name.find(marker)
    if pos < 0:
        return None
    start = pos + len(marker)
    end = name.find("_", start)
    token = name[start:] if end < 0 else name[start:end]
    if not token.isdigit():
        return None
    return int(token)


def build_channel_importance_timeline_curves(
    feature_importances: np.ndarray,
    feature_names: list[str],
    channel_names: list[str],
    observation_window_size: int,
    resolution_per_minute: int = 1,
) -> dict[str, np.ndarray]:
    """Map selected feature importances onto a timeline by channel.

    resolution_per_minute controls plot-bin density (e.g., 4 => 0.25-minute bins).
    """
    importances = np.asarray(feature_importances, dtype=np.float64).ravel()
    if importances.size != len(feature_names):
        raise ValueError(
            "Feature importance and feature name lengths do not match: "
            f"{importances.size} vs {len(feature_names)}."
        )

    obs = int(observation_window_size)
    res = max(1, int(resolution_per_minute))
    total_bins = int(obs * res)
    curves: dict[str, np.ndarray] = {
        str(channel): np.zeros(total_bins, dtype=np.float64) for channel in channel_names
    }
    for name, value in zip(feature_names, importances.tolist()):
        channel = channel_name_from_feature_name(name, channel_names)
        window = window_size_from_feature_name(name)
        slice_idx = slice_index_from_feature_name(name)
        if channel is None or window is None or slice_idx is None:
            continue

        start = int(slice_idx) * int(window)
        end = min(start + int(window), obs)
        if start < 0 or end <= start or start >= obs:
            continue
        start_bin = int(start * res)
        end_bin = int(end * res)
        if end_bin <= start_bin:
            continue
        width = end_bin - start_bin
        curves[str(channel)][start_bin:end_bin] += float(value) / float(width)

    return curves


def smooth_curve_gaussian(curve: np.ndarray, window_bins: int) -> np.ndarray:
    """Apply 1D Gaussian smoothing while preserving array length."""
    y = np.asarray(curve, dtype=np.float64).ravel()
    bins = int(window_bins)
    if bins <= 1 or y.size <= 2:
        return y.copy()

    radius = max(1, bins // 2)
    sigma = max(1.0, float(bins) / 3.0)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        return y.copy()
    kernel = kernel / kernel_sum

    y_pad = np.pad(y, (radius, radius), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")


def aggregate_feature_importance_by_channel(
    feature_importances: np.ndarray,
    feature_names: list[str],
    channel_names: list[str],
) -> tuple[dict[str, dict[str, float | int]], int]:
    importances = np.asarray(feature_importances, dtype=np.float64).ravel()
    if importances.size != len(feature_names):
        raise ValueError(
            "Feature importance and feature name lengths do not match: "
            f"{importances.size} vs {len(feature_names)}."
        )

    summary: dict[str, dict[str, float | int]] = {
        str(channel): {
            "importance_total": 0.0,
            "importance_mag": 0.0,
            "importance_phase": 0.0,
            "importance_share_total": 0.0,
            "importance_share_mag": 0.0,
            "importance_share_phase": 0.0,
            "feature_count_total": 0,
            "feature_count_mag": 0,
            "feature_count_phase": 0,
        }
        for channel in channel_names
    }

    unmatched_count = 0
    for name, value in zip(feature_names, importances.tolist()):
        matched_channel = channel_name_from_feature_name(name, channel_names)
        if matched_channel is None:
            unmatched_count += 1
            continue

        info = summary[matched_channel]
        importance = float(value)

        info["importance_total"] = float(info["importance_total"]) + importance
        info["feature_count_total"] = int(info["feature_count_total"]) + 1

        if "_mag_" in name:
            info["importance_mag"] = float(info["importance_mag"]) + importance
            info["feature_count_mag"] = int(info["feature_count_mag"]) + 1
        elif "_phase_" in name:
            info["importance_phase"] = float(info["importance_phase"]) + importance
            info["feature_count_phase"] = int(info["feature_count_phase"]) + 1

    total_importance = float(importances.sum())
    mag_total = float(sum(float(info["importance_mag"]) for info in summary.values()))
    phase_total = float(sum(float(info["importance_phase"]) for info in summary.values()))

    for info in summary.values():
        info["importance_share_total"] = (
            float(info["importance_total"]) / total_importance if total_importance > 0.0 else 0.0
        )
        info["importance_share_mag"] = (
            float(info["importance_mag"]) / mag_total if mag_total > 0.0 else 0.0
        )
        info["importance_share_phase"] = (
            float(info["importance_phase"]) / phase_total if phase_total > 0.0 else 0.0
        )

    return summary, int(unmatched_count)


def normalize_feature_importance_by_relevant_window_feature_count(
    feature_importances: np.ndarray,
    feature_names: list[str],
    channel_names: list[str],
) -> np.ndarray:
    """Normalize by (#selected features in the same window and same channel)."""
    importances = np.asarray(feature_importances, dtype=np.float64).ravel()
    if importances.size != len(feature_names):
        raise ValueError(
            "Feature importance and feature name lengths do not match: "
            f"{importances.size} vs {len(feature_names)}."
        )

    counts: dict[tuple[int, str], int] = {}
    for name in feature_names:
        window = window_size_from_feature_name(name)
        channel = channel_name_from_feature_name(name, channel_names)
        if window is None or channel is None:
            continue
        key = (int(window), str(channel))
        counts[key] = int(counts.get(key, 0)) + 1

    normalized = importances.copy()
    for idx, name in enumerate(feature_names):
        window = window_size_from_feature_name(name)
        channel = channel_name_from_feature_name(name, channel_names)
        if window is None or channel is None:
            continue
        denom = int(counts.get((int(window), str(channel)), 0))
        if denom > 0:
            normalized[idx] /= float(denom)

    return normalized


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
        ax.set_xlabel("Global top-% selected after window concatenation")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def normalize_feature_importance_by_window(
    feature_importances: np.ndarray,
    windows: list[int],
    selected_idx_by_window: dict[int, list[int]],
) -> np.ndarray:
    """Normalize concatenated importances by selected feature count in each window block."""
    importances = np.asarray(feature_importances, dtype=np.float64).ravel()
    normalized = importances.copy()

    start = 0
    for window in windows:
        count = int(len(selected_idx_by_window.get(int(window), [])))
        if count <= 0:
            continue
        end = start + count
        if end > normalized.size:
            raise ValueError(
                "Window-selection metadata exceeds feature-importance length: "
                f"window={window}, end={end}, total={normalized.size}."
            )
        normalized[start:end] /= float(count)
        start = end

    if start != normalized.size:
        raise ValueError(
            "Window-selection metadata does not match feature-importance length: "
            f"consumed={start}, total={normalized.size}."
        )

    return normalized


def plot_feature_importance_line_grid(
    model_artifacts: list[dict],
    results_df: pd.DataFrame,
    forecast_lags: list[int],
    top_percents: list[float],
    output_path: Path,
    normalize_by_window: bool = False,
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

    normalized_ymax = 0.0
    if normalize_by_window:
        for artifact in artifact_map.values():
            base_importances = np.asarray(artifact["model"].feature_importances_, dtype=np.float64).ravel()
            windows = [int(w) for w in artifact.get("windows", sorted(artifact["selected_feature_indices_by_window"].keys()))]
            selected_idx_by_window = artifact["selected_feature_indices_by_window"]
            norm_vals = normalize_feature_importance_by_window(
                feature_importances=base_importances,
                windows=windows,
                selected_idx_by_window=selected_idx_by_window,
            )
            finite_vals = norm_vals[np.isfinite(norm_vals)]
            if finite_vals.size > 0:
                normalized_ymax = max(normalized_ymax, float(np.max(finite_vals)))
        normalized_ymax = max(normalized_ymax, 1.0e-12)

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
                # Add light, non-overlapping window shading
                ax.axvspan(
                    start,
                    end,
                    color=window_colors[i % len(window_colors)],
                    alpha=0.18,
                    linewidth=0,
                    zorder=0,
                )
                if start > 0:
                    ax.axvline(start, color="#6f6f6f", linewidth=0.6, alpha=0.6, zorder=1)
                start = end

            plot_values = importances
            if normalize_by_window:
                plot_values = normalize_feature_importance_by_window(
                    feature_importances=importances,
                    windows=windows,
                    selected_idx_by_window=selected_idx_by_window,
                )

            ax.plot(x, plot_values, color="#1f77b4", linewidth=0.9)
            if normalize_by_window:
                ax.set_ylim(0.0, normalized_ymax * 1.08)
            else:
                ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, max(1, int(importances.size - 1)))
            ax.grid(True, axis="y", alpha=0.2)
            ax.tick_params(labelsize=7)

            window_counts = [int(len(selected_idx_by_window.get(int(window), []))) for window in windows]
            counts_label = " ".join([f"w{int(window)}:{count}" for window, count in zip(windows, window_counts)])
            ax.set_title(
                f"top {top_key}% | n={int(importances.size)} | {counts_label}",
                fontsize=9,
            )
            # shared labels handled at figure level
            if c == 0:
                ax.set_ylabel("")
            if r == nrows - 1:
                ax.set_xlabel("")

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

    line_label = (
        "normalized importance (/selected_features_in_window)"
        if normalize_by_window
        else "feature importance"
    )
    legend_handles = [Line2D([0], [0], color="#1f77b4", lw=1.2, label=line_label)]
    for i, window in enumerate(example_windows):
        legend_handles.append(
            Patch(facecolor=window_colors[i % len(window_colors)], edgecolor="none", alpha=0.35, label=f"window {window}")
        )

    if normalize_by_window:
        title = "Normalized feature-importance lines (importance / #selected-features in source window)"
        y_label = "importance / #selected-features in window"
    else:
        title = "Feature importance lines (5x5) with window-segment shading"
        y_label = "importance"
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.supxlabel("feature idx", fontsize=10)
    fig.supylabel(y_label, fontsize=10)
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=max(2, len(legend_handles)), frameon=False, fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_topk_feature_importance_by_window(
    model_artifacts: list[dict],
    selection_df: pd.DataFrame,
    forecast_lags: list[int],
    top_percents: list[float],
    output_dir: Path,
    observation_window_size: int,
    filename_suffix: str | None = None,
):
    """Plot top-k feature importances laid out per window (one subplot per window).

    For each (lag, top%) artifact we rebuild a dense importance vector of length
    `total_features_for_window`, filling non-selected positions with zeros. Values
    are normalized by the total feature count (features_per_window * n_slices).
    The resulting arrays are plotted per window; rows correspond to window sizes
    and columns to top-% selections for the given lag. Slice boundaries are shown
    as dotted vertical lines.
    """

    if not model_artifacts:
        return []

    # Map: (lag, top_key) -> artifact
    artifact_map: dict[tuple[int, int], dict] = {}
    for artifact in model_artifacts:
        artifact_map[(int(artifact["forecast_lag_min"]), int(artifact["top_percent_key"]))] = artifact

    lag_order = [int(l) for l in forecast_lags]
    top_keys = [int(percent_key(p)) for p in top_percents]
    if not lag_order or not top_keys:
        return []

    # Derive window -> total_feature_count per lag from selection_df columns
    window_total_cols = [c for c in selection_df.columns if c.startswith("window") and c.endswith("_total_features")]
    window_counts_by_lag: dict[int, dict[int, int]] = {}
    for lag in lag_order:
        rows = selection_df[selection_df["forecast_lag_min"] == int(lag)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        window_counts_by_lag[int(lag)] = {
            int(col[len("window") :].split("_")[0]): int(row[col]) for col in window_total_cols
        }

    example_artifact = next(iter(model_artifacts))
    windows = [int(w) for w in example_artifact.get("windows", sorted(example_artifact["selected_feature_indices_by_window"].keys()))]
    n_slices_by_window = {int(w): max(1, int(observation_window_size // int(w))) for w in windows}

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for lag in lag_order:
        # Prepare canvas: rows=windows, cols=top% keys
        nrows, ncols = len(windows), len(top_keys)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 1.9 * nrows), sharex=False, sharey=False)
        axes = np.asarray(axes).reshape(nrows, ncols)

        for c, top_key in enumerate(top_keys):
            artifact = artifact_map.get((int(lag), int(top_key)))
            if artifact is None:
                for r in range(nrows):
                    axes[r, c].axis("off")
                continue

            importances = np.asarray(artifact["model"].feature_importances_, dtype=np.float64).ravel()
            selected_idx_by_window = artifact["selected_feature_indices_by_window"]
            win_feature_counts = window_counts_by_lag.get(int(lag), {})
            feature_totals_override = {45: 1472, 90: 1472, 180: 1456, 360: 1448}

            # Model importances are ordered as concatenation of per-window selected indices (sorted).
            offset = 0
            for r, window in enumerate(windows):
                ax = axes[r, c]
                local_idx = [int(i) for i in selected_idx_by_window.get(int(window), [])]
                count = len(local_idx)
                if count > 0:
                    segment = importances[offset : offset + count]
                else:
                    segment = np.asarray([], dtype=np.float64)
                offset += count

                total_count = int(win_feature_counts.get(int(window), 0))
                if int(window) in feature_totals_override:
                    total_count = feature_totals_override[int(window)]
                elif total_count <= 0:
                    total_count = int(max(local_idx) + 1) if local_idx else 0

                values = np.zeros(total_count, dtype=np.float64)
                if count > 0:
                    values[np.asarray(local_idx, dtype=np.int64)] = segment

                if values.size == 0:
                    ax.axis("off")
                    continue

                x = np.arange(values.size, dtype=np.int64)
                ax.plot(x, values, color="#1f77b4", linewidth=0.8)
                ax.set_xlim(0, max(1, values.size - 1))
                ax.grid(True, axis="y", alpha=0.2)
                ax.tick_params(labelsize=7)

                # Add slice separators
                n_slices = n_slices_by_window.get(int(window), 1)
                per_slice = int(total_count / n_slices) if n_slices > 0 else total_count
                for slice_idx in range(1, n_slices):
                    boundary = slice_idx * per_slice
                    ax.axvline(boundary, color="#9a9a9a", linestyle=":", linewidth=0.7, alpha=0.8)

                title = f"lag{int(lag)} | w{int(window)} | top{top_key}% | sel={count}"
                ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel("importance", fontsize=8)
                if r == nrows - 1:
                    ax.set_xlabel("feature idx", fontsize=8)

            # sanity: ensure we consumed importances for present windows
            if offset != importances.size:
                warnings.warn(
                    f"Artifact lag={lag} top={top_key}%: expected {importances.size} importance entries, consumed {offset}."
                )

        suffix = f"_{filename_suffix}" if filename_suffix else ""
        save_path = output_dir / f"experiment15_topk_importance_by_window_lag{int(lag)}{suffix}.png"
        fig.suptitle(
            f"Lag {int(lag)} – top-k feature importances per window (zeros = not selected)",
            fontsize=12,
            y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

    return saved_paths


def plot_feature_importance_individual_models_by_lag(
    model_artifacts: list[dict],
    results_df: pd.DataFrame,
    forecast_lags: list[int],
    top_percents: list[float],
    output_dir: Path,
    normalize_by_window: bool = False,
    filename_suffix: str | None = None,
) -> list[Path]:
    if not model_artifacts:
        return []

    lag_order = [int(l) for l in forecast_lags]
    top_keys = [int(percent_key(p)) for p in top_percents]
    if not lag_order or not top_keys:
        return []

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

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    window_colors = ["#dbe9ff", "#ffe8d6", "#e2f3e2", "#f0e0ff"]

    for lag in lag_order:
        lag_artifacts = [artifact_map.get((int(lag), int(top_key))) for top_key in top_keys]
        if all(artifact is None for artifact in lag_artifacts):
            continue

        n_panels = len(top_keys)
        ncols = 3 if n_panels >= 3 else n_panels
        nrows = int(math.ceil(float(n_panels) / float(ncols)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 2.9 * nrows), sharey=True)
        axes = np.asarray(axes).reshape(nrows, ncols)

        normalized_ymax = 0.0
        if normalize_by_window:
            for artifact in lag_artifacts:
                if artifact is None:
                    continue
                base_importances = np.asarray(artifact["model"].feature_importances_, dtype=np.float64).ravel()
                windows = [
                    int(w)
                    for w in artifact.get("windows", sorted(artifact["selected_feature_indices_by_window"].keys()))
                ]
                selected_idx_by_window = artifact["selected_feature_indices_by_window"]
                norm_vals = normalize_feature_importance_by_window(
                    feature_importances=base_importances,
                    windows=windows,
                    selected_idx_by_window=selected_idx_by_window,
                )
                finite_vals = norm_vals[np.isfinite(norm_vals)]
                if finite_vals.size > 0:
                    normalized_ymax = max(normalized_ymax, float(np.max(finite_vals)))
            normalized_ymax = max(normalized_ymax, 1.0e-12)

        for idx, top_key in enumerate(top_keys):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            artifact = artifact_map.get((int(lag), int(top_key)))
            if artifact is None:
                ax.axis("off")
                continue

            importances = np.asarray(artifact["model"].feature_importances_, dtype=np.float64).ravel()
            x = np.arange(importances.size, dtype=np.int64)
            windows = [
                int(w)
                for w in artifact.get("windows", sorted(artifact["selected_feature_indices_by_window"].keys()))
            ]
            selected_idx_by_window = artifact["selected_feature_indices_by_window"]

            start = 0
            for i, window in enumerate(windows):
                count = int(len(selected_idx_by_window.get(int(window), [])))
                if count <= 0:
                    continue
                end = start + count
                ax.axvspan(
                    start,
                    end,
                    color=window_colors[i % len(window_colors)],
                    alpha=0.18,
                    linewidth=0,
                    zorder=0,
                )
                if start > 0:
                    ax.axvline(start, color="#6f6f6f", linewidth=0.6, alpha=0.6, zorder=1)
                start = end

            plot_values = importances
            if normalize_by_window:
                plot_values = normalize_feature_importance_by_window(
                    feature_importances=importances,
                    windows=windows,
                    selected_idx_by_window=selected_idx_by_window,
                )

            ax.plot(x, plot_values, color="#1f77b4", linewidth=0.9)
            if normalize_by_window:
                ax.set_ylim(0.0, normalized_ymax * 1.08)
            else:
                ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, max(1, int(importances.size - 1)))
            ax.grid(True, axis="y", alpha=0.2)
            ax.tick_params(labelsize=7)
            window_counts = [int(len(selected_idx_by_window.get(int(window), []))) for window in windows]
            counts_label = ", ".join([f"w{int(window)}:{count}" for window, count in zip(windows, window_counts)])
            ax.set_title(
                f"lag {lag} | top {top_key}% | n={int(importances.size)} | {counts_label}",
                fontsize=9,
            )
            # shared labels handled at figure level
            if c == 0:
                ax.set_ylabel("")
            if r == nrows - 1:
                ax.set_xlabel("")

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

        for idx in range(n_panels, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].axis("off")

        line_label = (
            "normalized importance (/selected_features_in_window)"
            if normalize_by_window
            else "feature importance"
        )
        legend_handles = [Line2D([0], [0], color="#1f77b4", lw=1.2, label=line_label)]
        example_artifact = next((a for a in lag_artifacts if a is not None), None)
        example_windows = []
        if example_artifact is not None:
            example_windows = [int(w) for w in example_artifact.get("windows", [])]
        for i, window in enumerate(example_windows):
            legend_handles.append(
                Patch(
                    facecolor=window_colors[i % len(window_colors)],
                    edgecolor="none",
                    alpha=0.35,
                    label=f"window {window}",
                )
            )

        suffix = f"_{filename_suffix}" if filename_suffix else ""
        if normalize_by_window:
            title = (
                f"Experiment 15: Lag {lag} feature importance by model "
                "(normalized by selected-feature count per window)"
            )
            file_name = (
                f"experiment15_lag{int(lag)}_feature_importance_by_model_"
                f"normalized_by_window_feature_count{suffix}.png"
            )
        else:
            title = f"Lag {lag} feature importance by model"
            file_name = f"experiment15_lag{int(lag)}_feature_importance_by_model{suffix}.png"

        fig.suptitle(title, fontsize=12, y=0.985)
        fig.supxlabel("feature idx", fontsize=10)
        fig.supylabel(
            "importance / #selected-features in window" if normalize_by_window else "importance",
            fontsize=10,
        )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=max(2, len(legend_handles)),
            frameon=False,
            fontsize=9,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        save_path = output_dir / file_name
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close()
        saved_paths.append(save_path)

    return saved_paths


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
    ax.set_title("Experiment 15: #features needed to exceed 50% cumulative importance")
    ax.set_xlabel("Global top-% selected after window concatenation")
    ax.set_ylabel("Feature count to exceed 50% cumulative importance")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(len(lags), 5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_feature_importance_by_channel(
    channel_importance_df: pd.DataFrame,
    output_path: Path,
    channel_order: list[str] | None = None,
    importance_col: str = "importance_share_total_norm_by_window_feature_count",
):
    if len(channel_importance_df) == 0:
        return

    plot_df = channel_importance_df.copy()
    if "channel_name" not in plot_df.columns:
        return
    if importance_col not in plot_df.columns:
        raise ValueError(
            f"Requested importance column '{importance_col}' not found in channel_importance_df."
        )

    if channel_order:
        ordered_channels = [str(ch) for ch in channel_order if str(ch) in set(plot_df["channel_name"].astype(str))]
    else:
        ordered_channels = sorted(plot_df["channel_name"].astype(str).unique().tolist())

    if not ordered_channels:
        return

    top_percents = sorted(plot_df["top_percent"].astype(float).unique().tolist())
    lags = sorted(plot_df["forecast_lag_min"].astype(int).unique().tolist())
    if not top_percents or not lags:
        return

    x = np.arange(len(top_percents), dtype=np.float64)
    x_labels = [f"{int(round(p * 100.0))}%" for p in top_percents]

    n_channels = len(ordered_channels)
    ncols = 2 if n_channels > 1 else 1
    nrows = int(math.ceil(float(n_channels) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 3.4 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(nrows, ncols)

    for idx, channel_name in enumerate(ordered_channels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ch_df = plot_df[plot_df["channel_name"].astype(str) == str(channel_name)].copy()
        ch_df = ch_df.sort_values(["forecast_lag_min", "top_percent"])

        for lag in lags:
            lag_df = ch_df[ch_df["forecast_lag_min"].astype(int) == int(lag)].sort_values("top_percent")
            if len(lag_df) == 0:
                continue
            ax.plot(
                x,
                lag_df[importance_col].astype(float).values,
                marker="o",
                linewidth=1.5,
                label=f"lag {lag}",
            )

        ax.set_title(f"{channel_short_name(channel_name)} channel")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0)
        if c == 0:
            ax.set_ylabel("normalized importance share")
        if r == nrows - 1:
            ax.set_xlabel("Global top-% selected after window concatenation")
        if idx == 0:
            ax.legend(frameon=False, fontsize=8, loc="best")

    for idx in range(n_channels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        "Experiment 15: Feature importance by channel "
        "(normalized by relevant selected-feature count per window)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_global_explainability_channel_panels(
    channel_importance_df: pd.DataFrame,
    output_path: Path,
    channel_order: list[str] | None = None,
    importance_col: str = "importance_share_total_norm_by_window_feature_count",
):
    if len(channel_importance_df) == 0:
        return

    required = {"forecast_lag_min", "top_percent", "channel_name", importance_col}
    if not required.issubset(set(channel_importance_df.columns)):
        return

    df = channel_importance_df.copy()
    available_channels = set(df["channel_name"].astype(str).unique().tolist())
    if channel_order:
        channels = [str(ch) for ch in channel_order if str(ch) in available_channels]
    else:
        channels = sorted(available_channels)
    if not channels:
        return

    top_percents = sorted(df["top_percent"].astype(float).unique().tolist())
    lags_desc = sorted(df["forecast_lag_min"].astype(int).unique().tolist(), reverse=True)
    if not top_percents or not lags_desc:
        return

    n_panels = len(top_percents)
    ncols = 3 if n_panels >= 3 else n_panels
    nrows = int(math.ceil(float(n_panels) / float(ncols)))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.4 * nrows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(nrows, ncols)

    color_map = {
        "p3": "#1f77b4",
        "p5": "#ff7f0e",
        "p7": "#2ca02c",
        "long_xray": "#d62728",
    }

    for idx, frac in enumerate(top_percents):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        frac_df = df[df["top_percent"].astype(float) == float(frac)].copy()

        for channel_name in channels:
            ch_df = frac_df[frac_df["channel_name"].astype(str) == str(channel_name)].copy()
            ch_df = ch_df.sort_values("forecast_lag_min", ascending=False)
            if len(ch_df) == 0:
                continue

            short = channel_short_name(channel_name)
            ax.plot(
                ch_df["forecast_lag_min"].astype(int).values,
                ch_df[importance_col].astype(float).values,
                marker="o",
                linewidth=1.8,
                color=color_map.get(short, "#7f7f7f"),
                label=short,
            )

        ax.set_title(f"Top {int(round(float(frac) * 100.0))}% selected")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(lags_desc)
        ax.grid(True, alpha=0.25)
        if c == 0:
            ax.set_ylabel("relative normalized importance share")
        if r == nrows - 1:
            ax.set_xlabel("forecast lag (minutes before onset)")
        ax.text(
            0.98,
            0.02,
            "closer to onset \u2192",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#3a3a3a",
        )

        if idx == 0:
            ax.legend(frameon=False, fontsize=8, loc="best")

    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        "Experiment 15: Global explainability by channel across lags and selection conditions "
        "(normalized by relevant selected-feature count per window)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_global_explainability_by_lag(
    channel_importance_df: pd.DataFrame,
    output_dir: Path,
    channel_order: list[str] | None = None,
    importance_col: str = "importance_share_total_norm_by_window_feature_count",
    filename_suffix: str | None = None,
) -> list[Path]:
    if len(channel_importance_df) == 0:
        return []

    required = {"forecast_lag_min", "top_percent", "channel_name", importance_col}
    if not required.issubset(set(channel_importance_df.columns)):
        return []

    df = channel_importance_df.copy()
    available_channels = set(df["channel_name"].astype(str).unique().tolist())
    if channel_order:
        channels = [str(ch) for ch in channel_order if str(ch) in available_channels]
    else:
        channels = sorted(available_channels)
    if not channels:
        return []

    top_percents = sorted(df["top_percent"].astype(float).unique().tolist())
    lags = sorted(df["forecast_lag_min"].astype(int).unique().tolist())
    if not top_percents or not lags:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    x = np.arange(len(top_percents), dtype=np.float64)
    x_labels = [f"{int(round(p * 100.0))}%" for p in top_percents]

    color_map = {
        "p3": "#1f77b4",
        "p5": "#ff7f0e",
        "p7": "#2ca02c",
        "long_xray": "#d62728",
    }

    suffix = f"_{filename_suffix}" if filename_suffix else ""
    for lag in lags:
        lag_df = df[df["forecast_lag_min"].astype(int) == int(lag)].copy()
        if len(lag_df) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for channel_name in channels:
            ch_df = lag_df[lag_df["channel_name"].astype(str) == str(channel_name)].copy()
            if len(ch_df) == 0:
                continue
            ch_df = ch_df.sort_values("top_percent")
            y_vals = [np.nan] * len(top_percents)
            lookup = {
                float(row["top_percent"]): float(row[importance_col]) for _, row in ch_df.iterrows()
            }
            for i, frac in enumerate(top_percents):
                if float(frac) in lookup:
                    y_vals[i] = lookup[float(frac)]

            short = channel_short_name(channel_name)
            ax.plot(
                x,
                y_vals,
                marker="o",
                linewidth=1.9,
                color=color_map.get(short, "#7f7f7f"),
                label=short,
            )

        ax.set_title(f"Lag {int(lag)} min: global explainability by channel")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Global top-% selected after window concatenation")
        ax.set_ylabel("relative normalized importance share")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9, loc="best")

        plt.tight_layout()
        save_path = output_dir / f"experiment15_global_explainability_lag{int(lag)}{suffix}.png"
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close()
        saved_paths.append(save_path)

    return saved_paths


def plot_global_explainability_timeline_by_lag(
    model_artifacts: list[dict],
    results_df: pd.DataFrame,
    forecast_lags: list[int],
    top_percents: list[float],
    output_dir: Path,
    channel_order: list[str],
    observation_window_size: int,
    normalize_by_relevant_window_feature_count: bool = True,
    timeline_resolution_per_minute: int = 12,
    smoothing_window_minutes: float = 6.0,
    filename_suffix: str | None = None,
) -> list[Path]:
    """For each lag, plot 360-minute channel timelines across all top-% settings."""
    if not model_artifacts:
        return []
    if not channel_order:
        return []

    lag_order = [int(l) for l in forecast_lags]
    top_keys = [int(percent_key(p)) for p in top_percents]
    if not lag_order or not top_keys:
        return []

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

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    obs = int(observation_window_size)
    res = max(1, int(timeline_resolution_per_minute))
    smooth_bins = max(1, int(round(float(smoothing_window_minutes) * float(res))))
    x = np.arange(int(obs * res), dtype=np.float64) / float(res)
    color_map = {
        "p3": "#1f77b4",
        "p5": "#ff7f0e",
        "p7": "#2ca02c",
        "long_xray": "#d62728",
    }

    for lag in lag_order:
        n_panels = len(top_keys)
        ncols = 3 if n_panels >= 3 else n_panels
        nrows = int(math.ceil(float(n_panels) / float(ncols)))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.5 * nrows), sharex=True, sharey=False)
        axes = np.asarray(axes).reshape(nrows, ncols)

        any_panel = False
        for i, top_key in enumerate(top_keys):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            artifact = artifact_map.get((int(lag), int(top_key)))
            if artifact is None:
                ax.axis("off")
                continue
            any_panel = True

            feature_names = [str(n) for n in artifact.get("feature_names", [])]
            model_importances = np.asarray(artifact["model"].feature_importances_, dtype=np.float64).ravel()
            if model_importances.size != len(feature_names):
                ax.axis("off")
                continue

            plot_importances = model_importances
            if normalize_by_relevant_window_feature_count:
                plot_importances = normalize_feature_importance_by_relevant_window_feature_count(
                    feature_importances=model_importances,
                    feature_names=feature_names,
                    channel_names=channel_order,
                )

            curves = build_channel_importance_timeline_curves(
                feature_importances=plot_importances,
                feature_names=feature_names,
                channel_names=channel_order,
                observation_window_size=obs,
                resolution_per_minute=res,
            )

            for ch_name in channel_order:
                short = channel_short_name(ch_name)
                raw_curve = curves.get(str(ch_name), np.zeros_like(x, dtype=np.float64))
                plot_curve = smooth_curve_gaussian(raw_curve, smooth_bins)
                ax.plot(
                    x,
                    plot_curve,
                    linewidth=1.3,
                    color=color_map.get(short, "#7f7f7f"),
                    label=f"parameter: {str(ch_name)}",
                )

            ax.set_title(f"top {top_key}%", fontsize=9)
            ax.set_xlim(0.0, max(1.0, float(obs)))
            ax.set_xticks(np.arange(0, obs + 1, 60))
            ax.set_xticks(np.arange(0, obs + 1, 15), minor=True)
            ax.grid(True, which="major", alpha=0.25)
            ax.grid(True, which="minor", alpha=0.12, linestyle=":")
            if c == 0:
                ax.set_ylabel("smoothed importance density", fontsize=8)
            if r == nrows - 1:
                ax.set_xlabel(f"time index (minutes, plot step={1.0 / float(res):.2f})", fontsize=8)
            if i == 0:
                ax.legend(frameon=False, fontsize=7, loc="upper left")

            val_test = summary_map.get((int(lag), int(top_key)))
            if val_test is not None:
                val_css, test_css = val_test
                ax.text(
                    0.98,
                    0.97,
                    f"V:{val_css:.3f} T:{test_css:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="right",
                    fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2},
                )

        for i in range(n_panels, nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r, c].axis("off")

        if not any_panel:
            plt.close(fig)
            continue

        if normalize_by_relevant_window_feature_count:
            title = (
                f"Experiment 15: Lag {int(lag)} global explainability timeline (0-{obs-1}) "
                "normalized by relevant selected-feature count "
                f"| gaussian smooth={float(smoothing_window_minutes):.2f} min"
            )
            file_name = (
                f"experiment15_global_explainability_timeline_lag{int(lag)}"
                f"_norm_relevant_window_feature_count{suffix}.png"
            )
        else:
            title = (
                f"Experiment 15: Lag {int(lag)} global explainability timeline (0-{obs-1}) "
                f"| gaussian smooth={float(smoothing_window_minutes):.2f} min"
            )
            file_name = f"experiment15_global_explainability_timeline_lag{int(lag)}{suffix}.png"

        fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = output_dir / file_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(save_path)

    return saved_paths


def run_lag_global_concatenated_top_percent_experiment(
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    positive_class = pick_positive_class(classes)
    print(f"Positive class for scoring: idx={positive_class}, label='{classes[positive_class]}'")

    results_rows: list[dict[str, float | int | str]] = []
    selection_rows: list[dict[str, float | int | str]] = []
    channel_importance_rows: list[dict[str, float | int | str]] = []
    model_artifacts: list[dict] = []

    for lag in forecast_lags:
        print(f"\n{'=' * 80}")
        print(f"Lag: {lag}")

        X_train_parts: list[np.ndarray] = []
        X_val_parts: list[np.ndarray] = []
        X_test_parts: list[np.ndarray] = []
        concat_feature_names_all: list[str] = []
        window_feature_counts: dict[int, int] = {}
        window_boundaries: dict[int, tuple[int, int]] = {}
        offset = 0

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
            total_features = int(X_train_flat.shape[1])
            if len(feature_names) != total_features:
                raise ValueError(
                    f"Feature name count mismatch for window {fft_window_size}: "
                    f"got {len(feature_names)} names, expected {total_features}."
                )

            X_train_parts.append(X_train_flat)
            X_val_parts.append(X_val_flat)
            X_test_parts.append(X_test_flat)
            concat_feature_names_all.extend(feature_names)

            window_feature_counts[int(fft_window_size)] = total_features
            window_boundaries[int(fft_window_size)] = (int(offset), int(offset + total_features))
            offset += total_features

        X_train_concat_all = np.concatenate(X_train_parts, axis=1)
        X_val_concat_all = np.concatenate(X_val_parts, axis=1)
        X_test_concat_all = np.concatenate(X_test_parts, axis=1)

        baseline_model = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state + int(lag) * 1000 + 17,
            n_jobs=-1,
        )
        baseline_model.fit(X_train_concat_all, y_train)
        baseline_importances = np.asarray(baseline_model.feature_importances_, dtype=np.float64)
        global_total_features = int(baseline_importances.size)
        baseline_half_count, baseline_half_reached = count_features_for_cumulative_importance(
            baseline_importances,
            threshold=0.50,
            strict=True,
        )

        for frac in top_percents:
            pkey = percent_key(frac)
            selected_idx_global, selected_count = select_top_percentage_indices(baseline_importances, frac)
            selected_idx_by_window: dict[int, list[int]] = {}
            ordered_global_parts: list[np.ndarray] = []

            for fft_window_size in fft_window_sizes:
                start, end = window_boundaries[int(fft_window_size)]
                in_window = selected_idx_global[
                    (selected_idx_global >= int(start)) & (selected_idx_global < int(end))
                ]
                in_window_sorted = np.sort(in_window)
                ordered_global_parts.append(in_window_sorted)

                local_idx = (in_window_sorted - int(start)).astype(np.int64)
                selected_idx_by_window[int(fft_window_size)] = [int(i) for i in local_idx.tolist()]

            selected_idx_ordered = np.concatenate(ordered_global_parts, axis=0).astype(np.int64)
            if selected_idx_ordered.size == 0:
                raise RuntimeError(
                    f"Global top-{pkey}% selection returned no features for lag={lag}."
                )

            X_train_concat = X_train_concat_all[:, selected_idx_ordered]
            X_val_concat = X_val_concat_all[:, selected_idx_ordered]
            X_test_concat = X_test_concat_all[:, selected_idx_ordered]
            concat_feature_names = [concat_feature_names_all[int(i)] for i in selected_idx_ordered.tolist()]

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
            channel_summary, channel_unmatched_count = aggregate_feature_importance_by_channel(
                feature_importances=model_importances,
                feature_names=concat_feature_names,
                channel_names=channel_names,
            )
            model_importances_norm = normalize_feature_importance_by_relevant_window_feature_count(
                feature_importances=model_importances,
                feature_names=concat_feature_names,
                channel_names=channel_names,
            )
            channel_summary_norm, _ = aggregate_feature_importance_by_channel(
                feature_importances=model_importances_norm,
                feature_names=concat_feature_names,
                channel_names=channel_names,
            )

            result_row: dict[str, float | int | str] = {
                "forecast_lag_min": int(lag),
                "top_percent": float(frac),
                "top_percent_label": f"{pkey}%",
                "windows_used": ",".join(str(w) for w in fft_window_sizes),
                "total_concat_features": int(X_train_concat.shape[1]),
                "global_total_features_before_selection": int(global_total_features),
                "baseline_feature_count_cumimp_gt50": int(baseline_half_count),
                "baseline_cumimp_reached_gt50": float(baseline_half_reached),
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
                "channel_unmatched_feature_count": int(channel_unmatched_count),
            }

            for fft_window_size in fft_window_sizes:
                win_total = int(window_feature_counts[int(fft_window_size)])
                win_selected = int(len(selected_idx_by_window[int(fft_window_size)]))
                result_row[f"window{int(fft_window_size)}_total_features"] = win_total
                result_row[f"window{int(fft_window_size)}_selected_features"] = win_selected

            for channel_name in channel_names:
                channel_key = channel_short_name(str(channel_name))
                channel_stats = channel_summary.get(str(channel_name), {})
                importance_total = float(channel_stats.get("importance_total", 0.0))
                importance_share_total = float(channel_stats.get("importance_share_total", 0.0))
                feature_count_total = int(channel_stats.get("feature_count_total", 0))
                importance_mag = float(channel_stats.get("importance_mag", 0.0))
                importance_phase = float(channel_stats.get("importance_phase", 0.0))
                feature_count_mag = int(channel_stats.get("feature_count_mag", 0))
                feature_count_phase = int(channel_stats.get("feature_count_phase", 0))
                importance_share_mag = float(channel_stats.get("importance_share_mag", 0.0))
                importance_share_phase = float(channel_stats.get("importance_share_phase", 0.0))
                channel_stats_norm = channel_summary_norm.get(str(channel_name), {})
                importance_total_norm = float(channel_stats_norm.get("importance_total", 0.0))
                importance_share_total_norm = float(channel_stats_norm.get("importance_share_total", 0.0))
                importance_mag_norm = float(channel_stats_norm.get("importance_mag", 0.0))
                importance_phase_norm = float(channel_stats_norm.get("importance_phase", 0.0))
                importance_share_mag_norm = float(channel_stats_norm.get("importance_share_mag", 0.0))
                importance_share_phase_norm = float(channel_stats_norm.get("importance_share_phase", 0.0))

                result_row[f"importance_sum_{channel_key}"] = importance_total
                result_row[f"importance_share_{channel_key}"] = importance_share_total
                result_row[f"feature_count_{channel_key}"] = feature_count_total
                result_row[f"importance_sum_norm_{channel_key}"] = importance_total_norm
                result_row[f"importance_share_norm_{channel_key}"] = importance_share_total_norm

                channel_importance_rows.append(
                    {
                        "forecast_lag_min": int(lag),
                        "top_percent": float(frac),
                        "top_percent_label": f"{pkey}%",
                        "channel_name": str(channel_name),
                        "channel_short_name": str(channel_key),
                        "importance_total": importance_total,
                        "importance_share_total": importance_share_total,
                        "importance_mag": importance_mag,
                        "importance_phase": importance_phase,
                        "importance_share_mag": importance_share_mag,
                        "importance_share_phase": importance_share_phase,
                        "importance_total_norm_by_window_feature_count": importance_total_norm,
                        "importance_share_total_norm_by_window_feature_count": importance_share_total_norm,
                        "importance_mag_norm_by_window_feature_count": importance_mag_norm,
                        "importance_phase_norm_by_window_feature_count": importance_phase_norm,
                        "importance_share_mag_norm_by_window_feature_count": importance_share_mag_norm,
                        "importance_share_phase_norm_by_window_feature_count": importance_share_phase_norm,
                        "feature_count_total": feature_count_total,
                        "feature_count_mag": feature_count_mag,
                        "feature_count_phase": feature_count_phase,
                        "unmatched_feature_count": int(channel_unmatched_count),
                        "val_css": float(val_metrics["css"]),
                        "test_css": float(test_metrics["css"]),
                    }
                )

            results_rows.append(result_row)
            selection_row: dict[str, float | int | str] = {
                "forecast_lag_min": int(lag),
                "top_percent": float(frac),
                "top_percent_label": f"{pkey}%",
                "selection_scope": "global_concat_all_windows",
                "total_features_in_global_model": int(global_total_features),
                "selected_feature_count": int(selected_count),
                "selected_feature_fraction": float(selected_count) / float(global_total_features),
                "global_model_feature_count_cumimp_gt50": int(baseline_half_count),
                "global_model_cumimp_reached_gt50": float(baseline_half_reached),
            }
            for fft_window_size in fft_window_sizes:
                selection_row[f"window{int(fft_window_size)}_total_features"] = int(
                    window_feature_counts[int(fft_window_size)]
                )
                selection_row[f"window{int(fft_window_size)}_selected_features"] = int(
                    len(selected_idx_by_window[int(fft_window_size)])
                )
            selection_rows.append(selection_row)

            model_artifacts.append(
                {
                    "forecast_lag_min": int(lag),
                    "top_percent": float(frac),
                    "top_percent_key": int(pkey),
                    "selection_scope": "global_concat_all_windows",
                    "windows": [int(w) for w in fft_window_sizes],
                    "model": model,
                    "feature_names": concat_feature_names,
                    "selected_feature_indices_by_window": selected_idx_by_window,
                    "selected_feature_indices_global_ordered": [int(i) for i in selected_idx_ordered.tolist()],
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

    return (
        pd.DataFrame(results_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(channel_importance_rows),
        model_artifacts,
    )


def save_model_artifacts(model_artifacts: list[dict], output_dir: Path, run_stamp: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for artifact in model_artifacts:
        lag = int(artifact["forecast_lag_min"])
        top_key = int(artifact["top_percent_key"])
        path = output_dir / f"experiment15_rf_lag{lag}_top{top_key}_{run_stamp}.pkl"
        with open(path, "wb") as f:
            pickle.dump(artifact, f)
        saved_paths.append(path)

    return saved_paths


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
    top_percents = [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]
    timeline_resolution_per_minute = 12
    timeline_smoothing_window_minutes = 6.0

    results_df, selection_df, channel_importance_df, model_artifacts = run_lag_global_concatenated_top_percent_experiment(
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

    output_dir = data_root / "reports" / "experiment15"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"experiment15_lag_models_global_toppercent_results_{stamp}.csv"
    selection_path = output_dir / f"experiment15_global_selection_details_{stamp}.csv"
    channel_importance_path = output_dir / f"experiment15_feature_importance_by_channel_{stamp}.csv"
    results_df.to_csv(results_path, index=False)
    selection_df.to_csv(selection_path, index=False)
    channel_importance_df.to_csv(channel_importance_path, index=False)

    models_dir = output_dir / "saved_models"
    saved_model_paths = save_model_artifacts(model_artifacts, models_dir, run_stamp=stamp)

    summary_plot_path = output_dir / f"experiment15_lag_toppercent_training_summary_{stamp}.png"
    plot_lag_toppercent_training_summary(
        results_df=results_df,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        output_path=summary_plot_path,
    )

    feature_importance_grid_path = output_dir / f"experiment15_feature_importance_line_grid_5x5_{stamp}.png"
    plot_feature_importance_line_grid(
        model_artifacts=model_artifacts,
        results_df=results_df,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        output_path=feature_importance_grid_path,
    )
    normalized_feature_importance_grid_path = (
        output_dir / f"experiment15_feature_importance_line_grid_5x5_normalized_by_window_feature_count_{stamp}.png"
    )
    plot_feature_importance_line_grid(
        model_artifacts=model_artifacts,
        results_df=results_df,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        output_path=normalized_feature_importance_grid_path,
        normalize_by_window=True,
    )
    per_lag_feature_importance_dir = output_dir / "feature_importance_by_lag"
    per_lag_feature_importance_paths = plot_feature_importance_individual_models_by_lag(
        model_artifacts=model_artifacts,
        results_df=results_df,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        output_dir=per_lag_feature_importance_dir,
        normalize_by_window=False,
        filename_suffix=stamp,
    )
    per_lag_feature_importance_norm_paths = plot_feature_importance_individual_models_by_lag(
        model_artifacts=model_artifacts,
        results_df=results_df,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        output_dir=per_lag_feature_importance_dir,
        normalize_by_window=True,
        filename_suffix=stamp,
    )

    topk_by_window_dir = output_dir / "topk_feature_importance_by_window"
    topk_by_window_paths = plot_topk_feature_importance_by_window(
        model_artifacts=model_artifacts,
        selection_df=selection_df,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        output_dir=topk_by_window_dir,
        observation_window_size=observation_window_size,
        filename_suffix=stamp,
    )

    feature_importance_channel_plot_path = output_dir / f"experiment15_feature_importance_by_channel_{stamp}.png"
    plot_feature_importance_by_channel(
        channel_importance_df=channel_importance_df,
        output_path=feature_importance_channel_plot_path,
        channel_order=target_channels,
        importance_col="importance_share_total_norm_by_window_feature_count",
    )

    global_explainability_by_lag_dir = output_dir / "global_explainability_by_lag"
    global_explainability_by_lag_paths = plot_global_explainability_by_lag(
        channel_importance_df=channel_importance_df,
        output_dir=global_explainability_by_lag_dir,
        channel_order=target_channels,
        importance_col="importance_share_total_norm_by_window_feature_count",
        filename_suffix=stamp,
    )
    global_explainability_timeline_dir = output_dir / "global_explainability_timeline_by_lag"
    global_explainability_timeline_paths = plot_global_explainability_timeline_by_lag(
        model_artifacts=model_artifacts,
        results_df=results_df,
        forecast_lags=forecast_lags,
        top_percents=top_percents,
        output_dir=global_explainability_timeline_dir,
        channel_order=target_channels,
        observation_window_size=observation_window_size,
        normalize_by_relevant_window_feature_count=True,
        timeline_resolution_per_minute=timeline_resolution_per_minute,
        smoothing_window_minutes=timeline_smoothing_window_minutes,
        filename_suffix=stamp,
    )

    half_importance_plot_path = output_dir / f"experiment15_features_required_cumimp_gt50_by_lag_toppercent_{stamp}.png"
    plot_features_needed_for_half_importance(results_df, half_importance_plot_path)

    print("\n" + "=" * 80)
    print("Experiment 15 (lag-wise models from global top-percent concat selection) finished.")
    print(f"Run timestamp: {stamp}")
    print(f"Saved results CSV: {results_path}")
    print(f"Saved lag-selection CSV: {selection_path}")
    print(f"Saved channel-separated feature-importance CSV: {channel_importance_path}")
    print(f"Saved lag/top% training summary plot: {summary_plot_path}")
    print(f"Saved 5x5 feature-importance line grid: {feature_importance_grid_path}")
    print(f"Saved normalized 5x5 feature-importance line grid: {normalized_feature_importance_grid_path}")
    print(f"Saved per-lag feature-importance model plots: {len(per_lag_feature_importance_paths)} files in {per_lag_feature_importance_dir}")
    print(
        "Saved per-lag normalized feature-importance model plots: "
        f"{len(per_lag_feature_importance_norm_paths)} files in {per_lag_feature_importance_dir}"
    )
    print(f"Saved channel-separated feature-importance plot: {feature_importance_channel_plot_path}")
    print(
        f"Saved global explainability per-lag plots: "
        f"{len(global_explainability_by_lag_paths)} files in {global_explainability_by_lag_dir}"
    )
    print(
        f"Saved global explainability timeline-by-lag plots: "
        f"{len(global_explainability_timeline_paths)} files in {global_explainability_timeline_dir}"
    )
    print(f"Saved >50% cumulative-importance count plot: {half_importance_plot_path}")
    print(f"Saved models: {len(saved_model_paths)} files in {models_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
