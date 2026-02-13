"""Experiment 8: Title: Final Model Training & Physics-Based Interpretability

Purpose:
To train the "Production" model using the optimal configuration identified in previous experiments (Window=90m, Top-25 Coefficients) and, crucially, to decode the model's internal logic into human-readable Solar Physics terms.

Key Steps:

Training & Persistence: Trains the TimeSeriesForestClassifier on the full training set and saves the model (.joblib) and metadata (.json) for future inference.

Semantic Mapping: It reverse-engineers the random forest's internal decision trees. It maps abstract "Interval Indices" back to physical Time Slices (e.g., "T-90m to T-0m") and Spectral Features (e.g., "Proton Flux Magnitude").

Timeline Visualization: It generates a "Density of Importance" timeline plot. This visualizes exactly when the critical precursor signals occur relative to the flare onset, proving whether the model detects early warnings or immediate triggers.

Outcome: A saved, high-accuracy model and a set of plots that explain physically why the model predicts a flare.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from aeon.classification.interval_based import TimeSeriesForestClassifier
from scipy.fft import rfft
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
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

    @staticmethod
    def _parse_event_time(filename: str) -> pd.Timestamp | pd.NaT:
        stem = Path(str(filename)).stem
        dt = pd.to_datetime(stem, format="%Y-%m-%d_%H-%M", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(stem, errors="coerce")
        return dt

    def __len__(self) -> int:
        return len(self.metadata)

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

        print(
            "Year-based split counts: "
            f"train={len(train_X)} (<=1992), "
            f"val={len(val_X)} (1992,2002], "
            f"test={len(test_X)} (2002,2018]"
        )
        return (
            (train_X, np.asarray(train_y, dtype=np.int32)),
            (val_X, np.asarray(val_y, dtype=np.int32)),
            (test_X, np.asarray(test_y, dtype=np.int32)),
        )


def extract_observation_window(
    series: np.ndarray,
    event_index: int,
    observation_window_size: int,
    forecast_lag: int,
) -> np.ndarray:
    input_end = event_index - forecast_lag
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
    """Build magnitude+phase FFT features using all coefficients."""
    obs_len = observation_window_size
    if obs_len % fft_window_size != 0:
        raise ValueError(f"FFT window size {fft_window_size} must divide observation length {obs_len}.")

    n_slices = obs_len // fft_window_size
    transformed = []
    max_coeffs = None

    for series in X:
        obs = extract_observation_window(
            series,
            event_index=event_index,
            observation_window_size=observation_window_size,
            forecast_lag=forecast_lag,
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

    n_samples, c_fft, _ = X_all.shape
    reshaped = X_all.reshape(n_samples, c_fft, n_slices, max_coeffs)
    selected = reshaped[:, :, :, :k]
    return selected.reshape(n_samples, c_fft, n_slices * k), k


def train_model_for_config(
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    fft_window_size: int,
    forecast_lag: int,
    coeff_setting: int | str,
    event_index: int,
    observation_window_size: int,
    random_state: int = 42,
):
    X_train_all, max_coeffs, n_slices, channels = build_fft_features_all(
        X_train,
        fft_window_size=fft_window_size,
        event_index=event_index,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
    )
    X_train_spec, k = select_top_k_coeffs(X_train_all, max_coeffs, n_slices, coeff_setting)

    model = TimeSeriesForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_spec, y_train)

    return model, {
        "max_available_coeffs": int(max_coeffs),
        "effective_coeffs": int(k),
        "n_slices": int(n_slices),
        "n_channels": int(channels),
        "input_shape_train": list(X_train_spec.shape),
    }


def build_model_input_for_inference(
    X: list[np.ndarray],
    fft_window_size: int,
    event_index: int,
    observation_window_size: int,
    forecast_lag: int,
    coeff_count: int,
) -> tuple[np.ndarray, dict[str, int]]:
    X_all, max_coeffs, n_slices, channels = build_fft_features_all(
        X,
        fft_window_size=fft_window_size,
        event_index=event_index,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
    )
    X_spec, k_used = select_top_k_coeffs(X_all, max_coeffs, n_slices, coeff_count)
    return X_spec, {
        "max_available_coeffs": int(max_coeffs),
        "effective_coeffs": int(k_used),
        "n_slices": int(n_slices),
        "n_channels": int(channels),
    }


def pick_positive_class(classes: np.ndarray) -> int:
    labels = [str(c).lower() for c in classes]
    priority_tokens = ["flare", "sep", "event", "positive", "yes"]

    for token in priority_tokens:
        for idx, label in enumerate(labels):
            if token in label:
                return idx

    return 1 if len(classes) > 1 else 0


def compute_solar_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> dict[str, float]:
    yt = (y_true == positive_class).astype(int)
    yp = (y_pred == positive_class).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tss = recall_pos + specificity - 1.0

    hss_num = 2.0 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = hss_num / hss_den if hss_den != 0 else 0.0

    css = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"tss": float(tss), "hss": float(hss), "css": float(css)}


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_class: int) -> dict[str, float]:
    solar = compute_solar_metrics(y_true, y_pred, positive_class)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_class, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=positive_class, average="binary", zero_division=0)),
        "tss": solar["tss"],
        "hss": solar["hss"],
        "css": solar["css"],
    }


def print_metrics(split_name: str, metrics: dict[str, float]) -> None:
    print(
        f"{split_name} metrics -> "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"tss={metrics['tss']:.4f}, "
        f"hss={metrics['hss']:.4f}, "
        f"css={metrics['css']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"recall={metrics['recall']:.4f}"
    )


def get_model_feature_importances(model) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        if importances.ndim == 1 and len(importances) > 0:
            return importances

    if hasattr(model, "estimators_"):
        child = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                imp = np.asarray(est.feature_importances_, dtype=float)
                if imp.ndim == 1 and len(imp) > 0:
                    child.append(imp)
        if child:
            min_len = min(len(x) for x in child)
            aligned = np.vstack([x[:min_len] for x in child])
            return aligned.mean(axis=0)

    return None


def build_feature_names(
    target_channels: list[str],
    fft_window_size: int,
    observation_window_size: int,
    forecast_lag: int,
    coeff_setting: int | str,
    event_index: int,
) -> list[str]:
    max_coeffs = fft_window_size // 2 + 1
    k = max_coeffs if coeff_setting == "all" else min(int(coeff_setting), max_coeffs)
    n_slices = observation_window_size // fft_window_size

    input_end = event_index - forecast_lag
    input_start = input_end - observation_window_size

    names: list[str] = []
    for component in ["mag", "phase"]:
        for ch in target_channels:
            for slice_idx in range(n_slices):
                win_start = input_start + slice_idx * fft_window_size
                win_end = win_start + fft_window_size
                for coeff_idx in range(k):
                    names.append(f"{ch}_fft@[{win_start}:{win_end}]_{component}_c{coeff_idx}")
    return names


def build_model_input_channel_names(target_channels: list[str]) -> list[str]:
    return [f"{ch}_mag" for ch in target_channels] + [f"{ch}_phase" for ch in target_channels]


def normalise_interval_stat_name(feature_obj) -> str:
    name = getattr(feature_obj, "__name__", feature_obj.__class__.__name__)
    aliases = {
        "row_mean": "mean",
        "row_std": "std",
        "row_slope": "slope",
        "row_numba_min": "min",
        "row_numba_max": "max",
        "row_median": "median",
        "row_quantile25": "q25",
        "row_quantile75": "q75",
    }
    if name in aliases:
        return aliases[name]
    if name.startswith("row_"):
        return name[4:]
    return name


def format_event_relative(minute_index: int, event_index: int) -> str:
    delta = int(minute_index) - int(event_index)
    if delta > 0:
        return f"T+{delta}m"
    if delta == 0:
        return "T-0m"
    return f"T{delta}m"


def build_tsf_semantic_feature_importance_table(
    model,
    target_channels: list[str],
    k_coeffs: int,
    n_slices: int,
    event_index: int,
    observation_window_size: int,
    forecast_lag: int,
    fft_window_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    input_channel_names = build_model_input_channel_names(target_channels)
    input_end = event_index - forecast_lag
    input_start = input_end - observation_window_size

    estimators = list(getattr(model, "estimators_", []))
    interval_sets = list(getattr(model, "intervals_", []))
    if len(estimators) == 0 or len(interval_sets) == 0:
        return pd.DataFrame()

    for est_idx in range(min(len(estimators), len(interval_sets))):
        est = estimators[est_idx]
        if not hasattr(est, "feature_importances_"):
            continue

        importances = np.asarray(est.feature_importances_, dtype=float).ravel()
        if importances.size == 0:
            continue

        feature_meta: list[dict[str, int | str]] = []
        reps = interval_sets[est_idx]
        for rep_idx, rep in enumerate(reps):
            rep_name = ""
            if hasattr(model, "_series_transformers") and rep_idx < len(model._series_transformers):
                st = model._series_transformers[rep_idx]
                rep_name = "" if st is None else st.__class__.__name__

            for interval in getattr(rep, "intervals_", []):
                if len(interval) == 5:
                    start, end, dim, feature_obj, dilation = interval
                elif len(interval) == 4:
                    start, end, dim, feature_obj = interval
                    dilation = 1
                else:
                    continue

                dim = int(dim)
                if rep_name == "" and 0 <= dim < len(input_channel_names):
                    dim_name = input_channel_names[dim]
                else:
                    prefix = rep_name if rep_name else f"rep{rep_idx}"
                    dim_name = f"{prefix}_dim{dim}"

                stat_name = normalise_interval_stat_name(feature_obj)

                start = int(start)
                end = int(end)
                dilation = int(dilation)
                start_idx = max(0, min(start, n_slices * k_coeffs - 1))
                end_idx = max(0, min(max(end - 1, start), n_slices * k_coeffs - 1))
                start_slice, start_coeff = divmod(start_idx, k_coeffs)
                end_slice, end_coeff = divmod(end_idx, k_coeffs)
                obs_window_start = input_start + start_slice * fft_window_size
                obs_window_end = input_start + (end_slice + 1) * fft_window_size
                rel_start = format_event_relative(obs_window_start, event_index)
                rel_end = format_event_relative(obs_window_end, event_index)
                slice_start_1b = int(start_slice) + 1
                slice_end_1b = int(end_slice) + 1

                feature_label = (
                    f"{dim_name}"
                    #f"_tsf_interval_[{start}:{end})"
                    #f"_fft_slice@[S{slice_start_1b}:S{slice_end_1b}]"
                    f"_@[{obs_window_start}:{obs_window_end}]"
                    #f"_time_rel@[{rel_start}:{rel_end}]"
                    f"_{stat_name}"
                )
                feature_meta.append(
                    {
                        "feature_label": feature_label,
                        "representation": rep_name if rep_name else "base",
                        "dim_index": dim,
                        "dim_name": dim_name,
                        "interval_start": start,
                        "interval_end": end,
                        "interval_dilation": dilation,
                        "start_slice": int(start_slice),
                        "end_slice": int(end_slice),
                        "start_slice_1based": int(slice_start_1b),
                        "end_slice_1based": int(slice_end_1b),
                        "start_coeff": int(start_coeff),
                        "end_coeff": int(end_coeff),
                        "observation_window_start_min": int(obs_window_start),
                        "observation_window_end_min": int(obs_window_end),
                        "observation_window_start_rel_min": int(obs_window_start - event_index),
                        "observation_window_end_rel_min": int(obs_window_end - event_index),
                        "observation_window_start_rel_label": rel_start,
                        "observation_window_end_rel_label": rel_end,
                        "stat_name": stat_name,
                    }
                )

        m = min(len(importances), len(feature_meta))
        for feat_idx in range(m):
            rows.append(
                {
                    "estimator_index": int(est_idx),
                    "internal_feature_index": int(feat_idx),
                    "importance": float(importances[feat_idx]),
                    **feature_meta[feat_idx],
                }
            )

    if not rows:
        return pd.DataFrame()

    per_tree_df = pd.DataFrame(rows)
    grouped = (
        per_tree_df.groupby(
            [
                "feature_label",
                "representation",
                "dim_index",
                "dim_name",
                "interval_start",
                "interval_end",
                "interval_dilation",
                "start_slice",
                "end_slice",
                "start_slice_1based",
                "end_slice_1based",
                "start_coeff",
                "end_coeff",
                "observation_window_start_min",
                "observation_window_end_min",
                "observation_window_start_rel_min",
                "observation_window_end_rel_min",
                "observation_window_start_rel_label",
                "observation_window_end_rel_label",
                "stat_name",
            ],
            as_index=False,
        )
        .agg(
            importance=("importance", "sum"),
            importance_mean=("importance", "mean"),
            occurrence_count=("importance", "count"),
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped.insert(0, "feature_index", np.arange(len(grouped), dtype=int))
    grouped["label_source"] = "tsf_interval_semantic"
    return grouped


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 30,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if importance_df.empty:
        plt.figure(figsize=(10, 3))
        plt.text(0.5, 0.5, "Feature importance not exposed by this model.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=220)
        plt.close()
        return

    top_df = importance_df.head(top_n)
    label_source = top_df["label_source"].iloc[0] if len(top_df) > 0 else "unknown"

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_df, x="importance", y="feature_label", orient="h", color="#1f77b4")
    plt.title(f"Top {top_n} Feature Importances ({label_source})")
    plt.xlabel("importance")
    plt.ylabel("feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_interval_importance_timeline(
    tsf_semantic_df: pd.DataFrame,
    output_path: Path,
    timeline_start: int,
    timeline_end: int,
    highlight_top_n: int = 20,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = {
        "observation_window_start_min",
        "observation_window_end_min",
        "importance",
        "feature_label",
    }
    if tsf_semantic_df.empty or not required_cols.issubset(set(tsf_semantic_df.columns)):
        plt.figure(figsize=(12, 3))
        plt.text(
            0.5,
            0.5,
            "Interval importance timeline unavailable.",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=220)
        plt.close()
        return

    x = np.arange(timeline_start, timeline_end + 1)
    curve = np.zeros_like(x, dtype=float)

    interval_df = tsf_semantic_df.copy()
    for row in interval_df.itertuples(index=False):
        start = max(int(row.observation_window_start_min), timeline_start)
        end = min(int(row.observation_window_end_min), timeline_end)
        if end <= start:
            continue
        width = end - start
        start_idx = start - timeline_start
        end_idx = end - timeline_start
        curve[start_idx:end_idx] += float(row.importance) / width

    fig, ax = plt.subplots(figsize=(16, 6))
    (line,) = ax.plot(
        x,
        curve,
        color="#1f77b4",
        linewidth=2.0,
        label="Aggregated interval importance density",
    )
    ax.fill_between(x, curve, color="#1f77b4", alpha=0.10)

    top_intervals = interval_df.sort_values("importance", ascending=False).head(highlight_top_n)
    for row in top_intervals.itertuples(index=False):
        start = max(int(row.observation_window_start_min), timeline_start)
        end = min(int(row.observation_window_end_min), timeline_end)
        if end <= start:
            continue
        ax.axvspan(start, end, color="#ff7f0e", alpha=0.08, linewidth=0)

    ax.set_xlim(timeline_start, timeline_end)
    ax.set_xlabel(f"Time index (minutes, {timeline_start}-{timeline_end})")
    ax.set_ylabel("Aggregated feature importance")
    ax.set_title("TSF Interval Importance Timeline (All Intervals)")
    ax.grid(alpha=0.20, linestyle="--")
    ax.set_xticks(np.arange(timeline_start, timeline_end + 1, 60))

    event_line = ax.axvline(
        timeline_end,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Event onset",
    )
    highlight_patch = Patch(
        facecolor="#ff7f0e",
        alpha=0.15,
        edgecolor="none",
        label=f"Top {highlight_top_n} interval bands",
    )
    ax.legend(handles=[line, highlight_patch, event_line], loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_feature_importance_table(
    importances: np.ndarray | None,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    base_cols = ["feature_index", "feature_label", "importance", "label_source"]
    if importances is None or len(importances) == 0:
        return pd.DataFrame(columns=base_cols)

    importances = np.asarray(importances, dtype=float)
    n = len(importances)

    if feature_names is not None and len(feature_names) == n:
        labels = list(feature_names)
        label_source = "semantic_input"
    else:
        labels = [f"internal_i{i}" for i in range(n)]
        label_source = "internal_index"

    df = pd.DataFrame(
        {
            "feature_index": np.arange(n, dtype=int),
            "feature_label": labels,
            "importance": importances,
            "label_source": label_source,
        }
    ).sort_values("importance", ascending=False)

    return df.reset_index(drop=True)


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

    # Given configuration
    event_onset_index = 720
    observation_window_size = 360
    fft_window_size = 90
    forecast_lag = 0
    coeff_setting: int | str = 25

    model, model_meta = train_model_for_config(
        X_train=X_train,
        y_train=y_train,
        fft_window_size=fft_window_size,
        forecast_lag=forecast_lag,
        coeff_setting=coeff_setting,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        random_state=42,
    )
    input_shape = tuple(model_meta["input_shape_train"])
    _, c_fft, t_fft = input_shape
    flattened_feature_count = c_fft * t_fft
    topk_active = str(coeff_setting) != "all" or (
        model_meta["effective_coeffs"] < model_meta["max_available_coeffs"]
    )
    print(
        "Model input dimensions: "
        f"shape={input_shape}, flattened_feature_count={flattened_feature_count}"
    )
    print(
        "Top-k details: "
        f"coeff_setting={coeff_setting}, "
        f"max_available_coeffs={model_meta['max_available_coeffs']}, "
        f"effective_coeffs={model_meta['effective_coeffs']}"
    )
    print(f"Top-k logic active: {'yes' if topk_active else 'no'}")

    positive_class = pick_positive_class(dataset.classes_)
    print(
        f"Positive class used for binary metrics: "
        f"index={positive_class}, label='{dataset.classes_[positive_class]}'"
    )

    train_spec, train_transform_meta = build_model_input_for_inference(
        X_train,
        fft_window_size=fft_window_size,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
        coeff_count=int(model_meta["effective_coeffs"]),
    )
    val_spec, val_transform_meta = build_model_input_for_inference(
        X_val,
        fft_window_size=fft_window_size,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
        coeff_count=int(model_meta["effective_coeffs"]),
    )
    test_spec, test_transform_meta = build_model_input_for_inference(
        X_test,
        fft_window_size=fft_window_size,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
        coeff_count=int(model_meta["effective_coeffs"]),
    )

    expected_shape_2d = tuple(model_meta["input_shape_train"][1:])
    for split_name, split_spec in [("train", train_spec), ("val", val_spec), ("test", test_spec)]:
        if tuple(split_spec.shape[1:]) != expected_shape_2d:
            raise ValueError(
                f"Feature shape mismatch for {split_name}: "
                f"got {tuple(split_spec.shape[1:])}, expected {expected_shape_2d}."
            )

    for split_name, split_meta in [
        ("train", train_transform_meta),
        ("val", val_transform_meta),
        ("test", test_transform_meta),
    ]:
        if split_meta["effective_coeffs"] != int(model_meta["effective_coeffs"]):
            raise ValueError(
                f"Effective coefficient mismatch for {split_name}: "
                f"got {split_meta['effective_coeffs']}, expected {model_meta['effective_coeffs']}."
            )

    y_pred_train = model.predict(train_spec)
    y_pred_val = model.predict(val_spec)
    y_pred_test = model.predict(test_spec)

    train_metrics = compute_all_metrics(y_train, y_pred_train, positive_class)
    val_metrics = compute_all_metrics(y_val, y_pred_val, positive_class)
    test_metrics = compute_all_metrics(y_test, y_pred_test, positive_class)

    print_metrics("Train", train_metrics)
    print_metrics("Validation", val_metrics)
    print_metrics("Test", test_metrics)

    output_dir = data_root / "reports" / "experiment8"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = output_dir / f"experiment8_tsf_model_{stamp}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved trained model: {model_path}")

    model_info = {
        "window_size": int(fft_window_size),
        "forecast_lag_min": int(forecast_lag),
        "coeff_setting": str(coeff_setting),
        "observation_window_size": int(observation_window_size),
        "event_onset_index": int(event_onset_index),
        "target_channels": target_channels,
        "class_names": [str(x) for x in dataset.classes_],
        "positive_class_index": int(positive_class),
        "positive_class_label": str(dataset.classes_[positive_class]),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        **model_meta,
    }
    model_info_path = output_dir / f"experiment8_tsf_model_metadata_{stamp}.json"
    with open(model_info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)
    print(f"Saved model metadata: {model_info_path}")

    importances = get_model_feature_importances(model)
    if importances is None or len(importances) == 0:
        print("Internal model feature importance not exposed by model.")
    else:
        print(f"Internal model feature dimension (importance length): {len(importances)}")

    feature_names = build_feature_names(
        target_channels=target_channels,
        fft_window_size=fft_window_size,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
        coeff_setting=coeff_setting,
        event_index=event_onset_index,
    )
    print(f"Semantic input feature-name count: {len(feature_names)}")
    if importances is not None and len(importances) > 0 and len(feature_names) != len(importances):
        print(
            "Importance/name length mismatch detected: "
            f"semantic_feature_count={len(feature_names)}, "
            f"internal_importance_count={len(importances)}. "
            "Using internal_i# labels for importance reporting."
        )

    tsf_semantic_df = build_tsf_semantic_feature_importance_table(
        model=model,
        target_channels=target_channels,
        k_coeffs=int(model_meta["effective_coeffs"]),
        n_slices=int(model_meta["n_slices"]),
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        forecast_lag=forecast_lag,
        fft_window_size=fft_window_size,
    )
    if tsf_semantic_df.empty:
        print("TSF semantic interval mapping unavailable from model internals.")
    else:
        print(
            "TSF semantic interval mapping: "
            f"generated {len(tsf_semantic_df)} aggregated semantic features."
        )

    importance_df = build_feature_importance_table(importances, feature_names=feature_names)
    if len(importance_df) > 0:
        top_n = min(30, len(importance_df))
        top_table = importance_df.head(top_n).copy()
        top_table.insert(0, "rank", np.arange(1, top_n + 1))
        top_table["importance_index_type"] = top_table["label_source"]
        print("\nTop feature names used for importance reporting:")
        print(
            top_table[["rank", "feature_label", "importance", "importance_index_type"]]
            .to_string(index=False)
        )
    else:
        print("Top feature names: unavailable (importance vector not exposed).")

    importance_table_path = output_dir / f"experiment8_feature_importance_table_{stamp}.csv"
    importance_df.to_csv(importance_table_path, index=False)
    print(f"Saved feature importance table: {importance_table_path}")

    fi_path = output_dir / f"experiment8_feature_importance_{stamp}.png"
    plot_feature_importance(importance_df, fi_path, top_n=30)
    print(f"Saved feature importance plot: {fi_path}")

    if not tsf_semantic_df.empty:
        tsf_top_n = min(30, len(tsf_semantic_df))
        tsf_top = tsf_semantic_df.head(tsf_top_n).copy()
        tsf_top.insert(0, "rank", np.arange(1, tsf_top_n + 1))
        print("\nTop TSF semantic interval features:")
        print(
            tsf_top[["rank", "feature_label", "importance", "occurrence_count"]]
            .to_string(index=False)
        )

        tsf_semantic_table_path = output_dir / f"experiment8_feature_importance_tsf_semantic_table_{stamp}.csv"
        tsf_semantic_df.to_csv(tsf_semantic_table_path, index=False)
        print(f"Saved TSF semantic importance table: {tsf_semantic_table_path}")

        tsf_semantic_plot_path = output_dir / f"experiment8_feature_importance_tsf_semantic_{stamp}.png"
        plot_feature_importance(tsf_semantic_df, tsf_semantic_plot_path, top_n=30)
        print(f"Saved TSF semantic feature importance plot: {tsf_semantic_plot_path}")

        tsf_timeline_plot_path = output_dir / f"experiment8_feature_importance_tsf_timeline_{stamp}.png"
        plot_interval_importance_timeline(
            tsf_semantic_df=tsf_semantic_df,
            output_path=tsf_timeline_plot_path,
            timeline_start=0,
            timeline_end=event_onset_index,
            highlight_top_n=20,
        )
        print(f"Saved TSF interval-importance timeline plot: {tsf_timeline_plot_path}")
    else:
        print("Skipped TSF semantic feature importance plot (mapping unavailable).")

if __name__ == "__main__":
    main()
