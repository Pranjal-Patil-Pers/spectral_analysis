from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_file_path(raw_dir: Path, filename: str) -> Path:
    name = str(filename)
    if not name.endswith(".csv"):
        name = f"{name}.csv"
    return raw_dir / name


def compute_duration_hours_ceil(series_df: pd.DataFrame, cadence_minutes: int = 1) -> int:
    """Compute series duration in hours using fixed cadence, rounded up to full hour."""
    if series_df.empty:
        return 0

    time_col = series_df.columns[0]
    ts = pd.to_datetime(series_df[time_col], errors="coerce")
    ts = ts.dropna()

    if ts.empty:
        return 0

    # With 1-minute cadence, include both endpoints: duration = (end-start) + cadence.
    duration_seconds = (ts.max() - ts.min()).total_seconds() + cadence_minutes * 60
    duration_hours = duration_seconds / 3600.0

    # Round up to full hour and keep at least 1 hour for non-empty valid series.
    return max(1, int(math.ceil(duration_hours)))


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

    if not labels_file.exists():
        raise FileNotFoundError(f"Missing labels file: '{labels_file}'")

    cadence_minutes = 1
    target_channels = ["p3_flux_ic", "p5_flux_ic", "p7_flux_ic", "long"]

    labels_df = pd.read_csv(labels_file)
    if "File" not in labels_df.columns:
        raise ValueError("labels file must contain a 'File' column")

    records = []

    for filename in labels_df["File"].tolist():
        file_path = build_file_path(raw_dir, filename)
        if not file_path.exists():
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            print(f"Skipping {file_path.name}: read failed ({exc})")
            continue

        missing_channels = [col for col in target_channels if col not in df.columns]
        if missing_channels:
            print(
                f"Skipping {file_path.name}: missing target channels {missing_channels}"
            )
            continue

        duration_hours_ceil = compute_duration_hours_ceil(df, cadence_minutes=cadence_minutes)
        if duration_hours_ceil == 0:
            print(f"Skipping {file_path.name}: no valid timestamps in first column")
            continue

        records.append(
            {
                "file": file_path.name,
                "length_hours_ceil": duration_hours_ceil,
                "num_rows": int(len(df)),
            }
        )

    if not records:
        raise RuntimeError("No valid time series files found for analysis.")

    result_df = pd.DataFrame(records)

    min_len = int(result_df["length_hours_ceil"].min())
    max_len = int(result_df["length_hours_ceil"].max())
    avg_len = int(math.ceil(result_df["length_hours_ceil"].mean()))
    min_row = result_df.loc[result_df["length_hours_ceil"].idxmin()]
    max_row = result_df.loc[result_df["length_hours_ceil"].idxmax()]

    print("Time Series Length Statistics (hours, rounded up)")
    print(f"Cadence: {cadence_minutes} minute per sample")
    print(f"Total files analyzed: {len(result_df)}")
    print(f"Min length (hours): {min_len}")
    print(f"Min length file: {min_row['file']} ({int(min_row['length_hours_ceil'])} hours)")
    print(f"Max length (hours): {max_len}")
    print(f"Max length file: {max_row['file']} ({int(max_row['length_hours_ceil'])} hours)")
    print(f"Avg length (hours): {avg_len}")

    output_dir = data_root / "reports"/"input_dataset_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_csv = output_dir / "timeseries_length_hours_summary.csv"
    result_df.to_csv(stats_csv, index=False)
    print(f"Saved per-file lengths to: {stats_csv}")

    freq_df = (
        result_df.groupby("length_hours_ceil", as_index=False)
        .size()
        .rename(columns={"size": "frequency"})
        .sort_values("length_hours_ceil")
    )
    freq_csv = output_dir / "timeseries_length_hours_frequency.csv"
    freq_df.to_csv(freq_csv, index=False)
    print(f"Saved length-frequency table to: {freq_csv}")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=result_df,
        x="length_hours_ceil",
        bins=np.arange(min_len - 0.5, max_len + 1.5, 1),
        discrete=True,
        color="#1f77b4",
        edgecolor="black",
    )
    plt.title("Histogram of Time Series Lengths (Hours, Rounded Up)")
    plt.xlabel("Length (hours)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    hist_path = output_dir / "timeseries_length_hours_histogram.png"
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved histogram to: {hist_path}")


if __name__ == "__main__":
    main()
