"""Experiment 13: 

experiment for plotting line plots per windowand lag setting for all coeff groups (all, 100, 50, 25, 10, 5) and selecting best config by val_css, val_tss, val_hss, val_f1 with test metrics as tie-breakers.

Run RandomForest on FFT magnitude+phase features using an
event-anchored window. For each forecast lag, data is extracted as
[event_index - lag - observation_window : event_index - lag], then
split into non-overlapping FFT windows across multiple sizes.

All coefficients are the size of all mag coeffcients generated per slice of the FFT, which is determined by the FFT window size. For example, a 360-minute window with 1-minute cadence generates 181 coefficients per channel (including DC), so "all" means using all 181 mag and 181 phase features per channel per slice.

Top-k is applied per-slice, selecting the first k magnitude and first k phase coefficients per channel (not k pairs), so effective_coeffs=k means using k magnitude and k phase features per channel per slice.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    plot_df = results_df.copy()
    plot_df["window_size"] = plot_df["window_size"].astype(int)
    plot_df["forecast_lag_min"] = plot_df["forecast_lag_min"].astype(int)
    plot_df["window_lag"] = plot_df.apply(
        lambda r: f"W{int(r['window_size'])}_lag{int(r['forecast_lag_min'])}",
        axis=1,
    )

    setups_df = (
        plot_df[["window_size", "forecast_lag_min", "window_lag"]]
        .drop_duplicates()
        .sort_values(["window_size", "forecast_lag_min"])
    )
    x_order = setups_df["window_lag"].tolist()

    coeff_levels = sorted(plot_df["effective_coeffs"].astype(int).unique().tolist())
    colors = plt.cm.get_cmap("tab10", max(1, len(coeff_levels)))

    fig, axes = plt.subplots(2, 4, figsize=(26, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        pivot = (
            plot_df.pivot_table(
                index="window_lag",
                columns="effective_coeffs",
                values=metric,
                aggfunc="mean",
            )
            .reindex(index=x_order, columns=coeff_levels)
        )

        for color_idx, coeff in enumerate(coeff_levels):
            ax.plot(
                x_order,
                pivot[coeff].values,
                marker="o",
                linewidth=1.5,
                markersize=4,
                color=colors(color_idx),
                label=f"k={coeff}",
            )

        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("window/lag setup")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=45)

    if len(axes) > len(metrics):
        axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(6, len(labels)),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 1.02),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def select_best(results_df: pd.DataFrame) -> pd.Series:
    return results_df.sort_values(
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
        ascending=False,
    ).iloc[0]


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
    coeff_options: list[int | str],
    event_index: int,
    observation_window_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    positive_class = pick_positive_class(classes)
    print(f"Positive class for scoring: idx={positive_class}, label='{classes[positive_class]}'")

    rows: list[dict[str, float | int | str]] = []

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

            for coeff in coeff_options:
                X_train_spec, k = select_top_k_coeffs(X_train_all, max_coeffs, n_slices, coeff)
                X_test_spec, _ = select_top_k_coeffs(X_test_all, max_coeffs, n_slices, coeff)
                X_val_spec, _ = select_top_k_coeffs(X_val_all, max_coeffs, n_slices, coeff)

                X_train_flat = X_train_spec.reshape(X_train_spec.shape[0], -1)
                X_val_flat = X_val_spec.reshape(X_val_spec.shape[0], -1)
                X_test_flat = X_test_spec.reshape(X_test_spec.shape[0], -1)

                print(
                    f"Running coeff={coeff} | all={max_coeffs} | using={k} "
                    f"(per-slice top-k, mag+phase) | rf_input_dim={X_train_flat.shape[1]}"
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

                rows.append(
                    {
                        "window_size": f"{int(fft_window_size):03}",
                        "forecast_lag_min": f"{int(lag):03}",
                        "input_start": input_start,
                        "input_end": input_end,
                        "coeff_setting": str(coeff),
                        "max_available_coeffs": max_coeffs,
                        "effective_coeffs": k,
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

                print(
                    " -> "
                    f"VAL: TSS={val_metrics['tss']:.4f}, HSS={val_metrics['hss']:.4f}, CSS={val_metrics['css']:.4f}, "
                    f"Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, "
                    f"Prec={val_metrics['precision']:.4f}, Rec={val_metrics['recall']:.4f} | "
                    f"TEST: TSS={test_metrics['tss']:.4f}, HSS={test_metrics['hss']:.4f}, CSS={test_metrics['css']:.4f}, "
                    f"Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, "
                    f"Prec={test_metrics['precision']:.4f}, Rec={test_metrics['recall']:.4f}"
                )

    if not rows:
        raise RuntimeError("No experiment result generated.")

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

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.get_splits()

    event_onset_index = 720
    observation_window_size = 360
    fft_window_sizes = [360, 180, 90, 45]
    forecast_lags = [5, 15, 30, 60, 120]
    coeff_options: list[int | str] = ["all", 100, 50, 25, 10, 5]

    results_df = run_experiments(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=dataset.classes_,
        fft_window_sizes=fft_window_sizes,
        forecast_lags=forecast_lags,
        coeff_options=coeff_options,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        random_state=42,
    )

    output_dir = data_root / "reports" / "experiment13"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"experiment13_results_{stamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    val_lineplot_path = output_dir / f"experiment13_val_metrics_lineplot_{stamp}.png"
    plot_metric_lineplots(results_df, val_lineplot_path, split_prefix="val")
    print(f"Saved validation line plot: {val_lineplot_path}")

    test_lineplot_path = output_dir / f"experiment13_test_metrics_lineplot_{stamp}.png"
    plot_metric_lineplots(results_df, test_lineplot_path, split_prefix="test")
    print(f"Saved test line plot: {test_lineplot_path}")

    best = select_best(results_df)
    best_path = output_dir / f"experiment13_best_config_{stamp}.csv"
    best.to_frame().T.to_csv(best_path, index=False)

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
  


if __name__ == "__main__":
    main()
