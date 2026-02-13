"""Experiment 7: FFT reconstruction analysis.

This script reconstructs time series after truncating FFT coefficients according to
specified settings, then compares original vs reconstructed signals.
It reports reconstruction error (MSE/MAE) over validation and test datasets, and
plots one sample per class for each configuration.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import irfft, rfft
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

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, str]:
        row = self.metadata.iloc[idx]
        df = pd.read_csv(row["full_path"])

        if self.feature_cols:
            series = df[self.feature_cols].values
        else:
            series = df.values

        return series.astype(np.float32), int(row["encoded_label"]), str(row[self.filename_col])

    def get_splits(self):
        train_X, train_y, train_id = [], [], []
        val_X, val_y, val_id = [], [], []
        test_X, test_y, test_id = [], [], []

        for idx in range(len(self)):
            row = self.metadata.iloc[idx]
            year = int(row["event_time"].year)

            try:
                series, label, sid = self[idx]
            except Exception as exc:
                print(f"Skipping sample index {idx}: {exc}")
                continue

            if year <= 1992:
                train_X.append(series)
                train_y.append(label)
                train_id.append(sid)
            elif 1992 < year <= 2002:
                val_X.append(series)
                val_y.append(label)
                val_id.append(sid)
            elif 2002 < year <= 2018:
                test_X.append(series)
                test_y.append(label)
                test_id.append(sid)

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
            (train_X, np.asarray(train_y, dtype=np.int32), train_id),
            (val_X, np.asarray(val_y, dtype=np.int32), val_id),
            (test_X, np.asarray(test_y, dtype=np.int32), test_id),
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


def reconstruct_with_topk(
    obs: np.ndarray,
    fft_window_size: int,
    coeff_setting: int | str,
) -> tuple[np.ndarray, int, int]:
    """Reconstruct a window using top-k FFT coefficients (complex => mag+phase)."""
    obs_len, _ = obs.shape
    if obs_len % fft_window_size != 0:
        raise ValueError(
            f"FFT window size {fft_window_size} must divide observation length {obs_len}."
        )

    n_slices = obs_len // fft_window_size
    reconstructed = np.zeros_like(obs, dtype=np.float32)

    max_coeffs = fft_window_size // 2 + 1
    if coeff_setting == "all":
        k = max_coeffs
    else:
        k = min(int(coeff_setting), max_coeffs)

    for s in range(n_slices):
        a = s * fft_window_size
        b = a + fft_window_size
        chunk = obs[a:b, :]

        fft_vals = rfft(chunk, axis=0)
        truncated = np.zeros_like(fft_vals)
        truncated[:k, :] = fft_vals[:k, :]

        recon = irfft(truncated, n=fft_window_size, axis=0)
        reconstructed[a:b, :] = recon.astype(np.float32)

    return reconstructed, max_coeffs, k


def compute_reconstruction_errors(
    X: list[np.ndarray],
    y: np.ndarray,
    ids: list[str],
    classes: np.ndarray,
    event_index: int,
    observation_window_size: int,
    fft_window_size: int,
    forecast_lag: int,
    coeff_setting: int | str,
) -> tuple[pd.DataFrame, int, int]:
    rows = []
    max_coeffs_ref = None
    k_ref = None

    for series, label, sid in zip(X, y, ids):
        obs = extract_observation_window(
            series,
            event_index=event_index,
            observation_window_size=observation_window_size,
            forecast_lag=forecast_lag,
        )
        recon, max_coeffs, k = reconstruct_with_topk(
            obs,
            fft_window_size=fft_window_size,
            coeff_setting=coeff_setting,
        )

        if max_coeffs_ref is None:
            max_coeffs_ref = max_coeffs
            k_ref = k

        mse = float(np.mean((obs - recon) ** 2))
        mae = float(np.mean(np.abs(obs - recon)))

        rows.append(
            {
                "sample_id": sid,
                "label": int(label),
                "label_name": str(classes[int(label)]),
                "mse": mse,
                "mae": mae,
            }
        )

    return pd.DataFrame(rows), int(max_coeffs_ref), int(k_ref)


def plot_one_sample_per_class(
    sample_error_df: pd.DataFrame,
    X: list[np.ndarray],
    ids: list[str],
    channels: list[str],
    event_index: int,
    observation_window_size: int,
    fft_window_size: int,
    forecast_lag: int,
    coeff_setting: int | str,
    output_path: Path,
):
    chosen_rows = []
    for cls in sorted(sample_error_df["label"].unique()):
        cls_df = sample_error_df[sample_error_df["label"] == cls].sort_values("mse")
        if len(cls_df) > 0:
            chosen_rows.append(cls_df.iloc[0])

    if not chosen_rows:
        return

    id_to_idx = {sid: i for i, sid in enumerate(ids)}
    fig, axes = plt.subplots(
        len(chosen_rows),
        len(channels),
        figsize=(4 * len(channels), 3 * len(chosen_rows)),
        squeeze=False,
    )

    for r, row in enumerate(chosen_rows):
        idx = id_to_idx[row["sample_id"]]
        obs = extract_observation_window(
            X[idx],
            event_index=event_index,
            observation_window_size=observation_window_size,
            forecast_lag=forecast_lag,
        )
        recon, _, k = reconstruct_with_topk(
            obs,
            fft_window_size=fft_window_size,
            coeff_setting=coeff_setting,
        )

        for c, channel in enumerate(channels):
            ax = axes[r][c]
            ax.plot(obs[:, c], label="original", linewidth=1.2)
            ax.plot(recon[:, c], label="reconstructed", linewidth=1.0)
            if r == 0:
                ax.set_title(channel)
            if c == 0:
                ax.set_ylabel(f"class={row['label_name']}\nMSE={row['mse']:.2e}")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        f"Original vs Reconstructed | W={fft_window_size}, lag={forecast_lag}, coeff={coeff_setting} (k={k})",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 0.98, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def run_reconstruction_experiment(
    X_val: list[np.ndarray],
    y_val: np.ndarray,
    val_ids: list[str],
    X_test: list[np.ndarray],
    y_test: np.ndarray,
    test_ids: list[str],
    classes: np.ndarray,
    channels: list[str],
    event_index: int,
    observation_window_size: int,
    fft_window_sizes: list[int],
    forecast_lags: list[int],
    coeff_options: list[int | str],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []

    for fft_window in fft_window_sizes:
        for lag in forecast_lags:
            for coeff in coeff_options:
                val_samples_df, max_coeffs, k = compute_reconstruction_errors(
                    X=X_val,
                    y=y_val,
                    ids=val_ids,
                    classes=classes,
                    event_index=event_index,
                    observation_window_size=observation_window_size,
                    fft_window_size=fft_window,
                    forecast_lag=lag,
                    coeff_setting=coeff,
                )
                test_samples_df, _, _ = compute_reconstruction_errors(
                    X=X_test,
                    y=y_test,
                    ids=test_ids,
                    classes=classes,
                    event_index=event_index,
                    observation_window_size=observation_window_size,
                    fft_window_size=fft_window,
                    forecast_lag=lag,
                    coeff_setting=coeff,
                )

                rows.append(
                    {
                        "window_size": fft_window,
                        "forecast_lag_min": lag,
                        "coeff_setting": str(coeff),
                        "max_available_coeffs": max_coeffs,
                        "effective_coeffs": k,
                        "val_mse": float(val_samples_df["mse"].mean()),
                        "val_mae": float(val_samples_df["mae"].mean()),
                        "test_mse": float(test_samples_df["mse"].mean()),
                        "test_mae": float(test_samples_df["mae"].mean()),
                    }
                )

                print(
                    f"window={fft_window}, lag={lag}, coeff={coeff}, all={max_coeffs}, using={k} "
                    f"| val_mse={rows[-1]['val_mse']:.4e}, test_mse={rows[-1]['test_mse']:.4e}"
                )

                plot_one_sample_per_class(
                    sample_error_df=val_samples_df,
                    X=X_val,
                    ids=val_ids,
                    channels=channels,
                    event_index=event_index,
                    observation_window_size=observation_window_size,
                    fft_window_size=fft_window,
                    forecast_lag=lag,
                    coeff_setting=coeff,
                    output_path=output_dir / f"val_recon_window{fft_window}_lag{lag}_coeff{coeff}.png",
                )
                plot_one_sample_per_class(
                    sample_error_df=test_samples_df,
                    X=X_test,
                    ids=test_ids,
                    channels=channels,
                    event_index=event_index,
                    observation_window_size=observation_window_size,
                    fft_window_size=fft_window,
                    forecast_lag=lag,
                    coeff_setting=coeff,
                    output_path=output_dir / f"test_recon_window{fft_window}_lag{lag}_coeff{coeff}.png",
                )

    if not rows:
        raise RuntimeError("No reconstruction results generated.")

    return pd.DataFrame(rows)


def plot_error_heatmap(results_df: pd.DataFrame, output_path: Path, split_prefix: str):
    metric = f"{split_prefix}_mse"
    coeff_order = ["all", "100", "50", "25", "10", "5"]

    plot_df = results_df.copy()
    plot_df["window_lag"] = plot_df.apply(
        lambda r: f"W{int(r['window_size'])}_L{int(r['forecast_lag_min'])}",
        axis=1,
    )

    pivot = (
        plot_df.pivot_table(index="window_lag", columns="coeff_setting", values=metric, aggfunc="mean")
        .reindex(columns=coeff_order)
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt=".2e", cmap="magma_r", cbar=True)
    plt.title(f"{split_prefix.upper()} Reconstruction MSE")
    plt.xlabel("coefficients")
    plt.ylabel("window/lag")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


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

    (_, _, _), (X_val, y_val, val_ids), (X_test, y_test, test_ids) = dataset.get_splits()

    event_onset_index = 720
    observation_window_size = 360
    fft_window_sizes = [360, 180, 90, 45]
    forecast_lags = [0, 5, 10, 60]
    coeff_options: list[int | str] = ["all", 100, 50, 25, 10, 5]

    output_dir = data_root / "reports" / "experiment7"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = run_reconstruction_experiment(
        X_val=X_val,
        y_val=y_val,
        val_ids=val_ids,
        X_test=X_test,
        y_test=y_test,
        test_ids=test_ids,
        classes=dataset.classes_,
        channels=target_channels,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        fft_window_sizes=fft_window_sizes,
        forecast_lags=forecast_lags,
        coeff_options=coeff_options,
        output_dir=output_dir,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"experiment7_reconstruction_errors_{stamp}.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSaved reconstruction summary: {summary_path}")

    val_heatmap = output_dir / f"experiment7_val_mse_heatmap_{stamp}.png"
    test_heatmap = output_dir / f"experiment7_test_mse_heatmap_{stamp}.png"
    plot_error_heatmap(results_df, val_heatmap, split_prefix="val")
    plot_error_heatmap(results_df, test_heatmap, split_prefix="test")
    print(f"Saved validation heatmap: {val_heatmap}")
    print(f"Saved testing heatmap: {test_heatmap}")


if __name__ == "__main__":
    main()
