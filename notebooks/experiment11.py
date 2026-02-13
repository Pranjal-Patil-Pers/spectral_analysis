"""Experiment 11: Run RandomForest on anchored shapelet-distance features.

For each forecast lag, data is extracted as:
  [event_index - lag - observation_window : event_index - lag]

For each configured shapelet length, anchored shapelets are sampled from train data,
distance features are computed for train/val/test, and the same metrics/heatmaps are
logged as prior experiments.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    def _parse_event_time(filename: str) -> pd.Timestamp | pd.NaT:
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


def build_observation_windows(
    X: list[np.ndarray],
    event_index: int,
    observation_window_size: int,
    forecast_lag: int,
) -> np.ndarray:
    """Build observation windows as (N, T, C)."""
    windows = []
    for series in X:
        windows.append(
            extract_observation_window(
                series,
                event_index=event_index,
                observation_window_size=observation_window_size,
                lag_minute=forecast_lag,
            )
        )
    return np.asarray(windows, dtype=np.float32)


def build_shapelet_bank(
    X_train_windows: np.ndarray,
    shapelet_length: int,
    random_state: int,
    max_samples_per_anchor: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sample anchored shapelets from train windows.

    Returns:
      shapelets: (S, L)
      shapelet_channels: (S,)
      shapelet_starts: (S,)
      n_anchors: number of non-overlapping anchor positions
    """
    N, T, C = X_train_windows.shape
    if shapelet_length > T:
        raise ValueError(f"shapelet_length={shapelet_length} exceeds window length T={T}.")
    if T % shapelet_length != 0:
        raise ValueError(f"shapelet_length={shapelet_length} must divide observation window T={T}.")

    anchor_starts = list(range(0, T, shapelet_length))
    rng = np.random.default_rng(random_state)

    shapelets: list[np.ndarray] = []
    shapelet_channels: list[int] = []
    shapelet_starts: list[int] = []

    for c in range(C):
        for s in anchor_starts:
            sample_idx = np.arange(N)
            rng.shuffle(sample_idx)
            chosen = sample_idx[: min(max_samples_per_anchor, N)]

            segs = X_train_windows[chosen, s : s + shapelet_length, c]
            segs = segs.astype(np.float32)
            seg_mean = segs.mean(axis=1, keepdims=True)
            seg_std = segs.std(axis=1, keepdims=True) + 1e-8
            segs = (segs - seg_mean) / seg_std

            for seg in segs:
                shapelets.append(seg)
                shapelet_channels.append(c)
                shapelet_starts.append(s)

    if len(shapelets) == 0:
        raise ValueError("No shapelets generated from training windows.")

    return (
        np.asarray(shapelets, dtype=np.float32),
        np.asarray(shapelet_channels, dtype=np.int32),
        np.asarray(shapelet_starts, dtype=np.int32),
        len(anchor_starts),
    )


def compute_shapelet_distance_features(
    X_windows: np.ndarray,
    shapelets: np.ndarray,
    shapelet_channels: np.ndarray,
    shapelet_starts: np.ndarray,
) -> np.ndarray:
    """Compute anchored z-normalized Euclidean distance features as (N, S)."""
    N, _, _ = X_windows.shape
    S, L = shapelets.shape
    feats = np.empty((N, S), dtype=np.float32)

    for j in range(S):
        c = int(shapelet_channels[j])
        s = int(shapelet_starts[j])
        subseq = X_windows[:, s : s + L, c].astype(np.float32)  # (N, L)
        sub_mean = subseq.mean(axis=1, keepdims=True)
        sub_std = subseq.std(axis=1, keepdims=True) + 1e-8
        subseq = (subseq - sub_mean) / sub_std
        diff = subseq - shapelets[j][None, :]
        feats[:, j] = np.sqrt(np.mean(diff * diff, axis=1))

    return feats


def rank_shapelets_by_train_separation(X_train_feats: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Rank shapelets by class-separation score (between-class / within-class variance)."""
    n_features = X_train_feats.shape[1]
    classes = np.unique(y_train)
    if len(classes) <= 1:
        return np.arange(n_features)

    class_means = []
    within = np.zeros(n_features, dtype=np.float64)
    for cls in classes:
        Xc = X_train_feats[y_train == cls]
        class_means.append(Xc.mean(axis=0))
        if len(Xc) > 1:
            within += Xc.var(axis=0)

    between = np.var(np.vstack(class_means), axis=0)
    score = between / (within + 1e-8)
    return np.argsort(score)[::-1]


def select_top_k_shapelets(X_all: np.ndarray, n_shapelets: int | str) -> tuple[np.ndarray, int]:
    if n_shapelets == "all":
        k = X_all.shape[1]
    else:
        k = min(int(n_shapelets), X_all.shape[1])
    return X_all[:, :k], k


def plot_metric_heatmaps(results_df: pd.DataFrame, output_path: Path, split_prefix: str):
    metrics = [
        f"{split_prefix}_tss",
        f"{split_prefix}_hss",
        f"{split_prefix}_css",
        f"{split_prefix}_accuracy",
        f"{split_prefix}_f1",
        f"{split_prefix}_precision",
        f"{split_prefix}_recall",
    ]
    coeff_order = ["all", "100", "50", "25", "10", "5"]

    plot_df = results_df.copy()
    plot_df["window_lag"] = plot_df.apply(
        lambda r: f"W{int(r['window_size'])}_H{int(r['forecast_lag_min'])}",
        axis=1,
    )

    fig, axes = plt.subplots(2, 4, figsize=(26, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        pivot = (
            plot_df.pivot_table(
                index="window_lag",
                columns="coeff_setting",
                values=metric,
                aggfunc="mean",
            )
            .reindex(columns=coeff_order)
        )

        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar=True)
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("shapelet count")
        ax.set_ylabel("window/lag")

    # Hide the unused subplot (8th pane for 7 metrics)
    if len(axes) > len(metrics):
        axes[-1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def select_best(results_df: pd.DataFrame) -> pd.Series:
    return results_df.sort_values(
        by=["val_tss", "val_hss", "val_f1", "test_tss", "test_hss", "test_f1"],
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
    shapelet_lengths: list[int],
    forecast_lags: list[int],
    shapelet_options: list[int | str],
    event_index: int,
    observation_window_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    positive_class = pick_positive_class(classes)
    print(f"Positive class for scoring: idx={positive_class}, label='{classes[positive_class]}'")

    rows: list[dict[str, float | int | str]] = []

    for shapelet_length in shapelet_lengths:
        for lag in forecast_lags:
            input_end = event_index - lag
            input_start = input_end - observation_window_size
            print(
                f"\n=== Shapelet length={shapelet_length} min | lag={lag} min "
                f"| Input window=[{input_start}:{input_end}] ==="
            )

            X_train_windows = build_observation_windows(
                X_train,
                event_index=event_index,
                observation_window_size=observation_window_size,
                forecast_lag=lag,
            )
            X_val_windows = build_observation_windows(
                X_val,
                event_index=event_index,
                observation_window_size=observation_window_size,
                forecast_lag=lag,
            )
            X_test_windows = build_observation_windows(
                X_test,
                event_index=event_index,
                observation_window_size=observation_window_size,
                forecast_lag=lag,
            )

            shapelets, shapelet_channels, shapelet_starts, n_anchors = build_shapelet_bank(
                X_train_windows,
                shapelet_length=shapelet_length,
                random_state=random_state + shapelet_length + lag,
            )

            X_train_all = compute_shapelet_distance_features(
                X_train_windows,
                shapelets=shapelets,
                shapelet_channels=shapelet_channels,
                shapelet_starts=shapelet_starts,
            )
            X_val_all = compute_shapelet_distance_features(
                X_val_windows,
                shapelets=shapelets,
                shapelet_channels=shapelet_channels,
                shapelet_starts=shapelet_starts,
            )
            X_test_all = compute_shapelet_distance_features(
                X_test_windows,
                shapelets=shapelets,
                shapelet_channels=shapelet_channels,
                shapelet_starts=shapelet_starts,
            )

            rank_idx = rank_shapelets_by_train_separation(X_train_all, y_train)
            X_train_all = X_train_all[:, rank_idx]
            X_val_all = X_val_all[:, rank_idx]
            X_test_all = X_test_all[:, rank_idx]

            max_shapelets = X_train_all.shape[1]
            print(
                f"all shapelets available for this setup: {max_shapelets} "
                f"[anchors={n_anchors}, channels={X_train_windows.shape[2]}]"
            )

            for shapelet_opt in shapelet_options:
                X_train_feats, k = select_top_k_shapelets(X_train_all, shapelet_opt)
                X_val_feats, _ = select_top_k_shapelets(X_val_all, shapelet_opt)
                X_test_feats, _ = select_top_k_shapelets(X_test_all, shapelet_opt)

                print(
                    f"Running shapelets={shapelet_opt} | all={max_shapelets} | using={k} "
                    f"| rf_input_dim={X_train_feats.shape[1]}"
                )

                model = RandomForestClassifier(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_train_feats, y_train)

                y_pred_train = model.predict(X_train_feats)
                y_pred_val = model.predict(X_val_feats)
                y_pred_test = model.predict(X_test_feats)

                train_metrics = compute_all_metrics(y_train, y_pred_train, positive_class)
                val_metrics = compute_all_metrics(y_val, y_pred_val, positive_class)
                test_metrics = compute_all_metrics(y_test, y_pred_test, positive_class)

                rows.append(
                    {
                        "window_size": shapelet_length,
                        "forecast_lag_min": lag,
                        "input_start": input_start,
                        "input_end": input_end,
                        "coeff_setting": str(shapelet_opt),
                        "max_available_coeffs": max_shapelets,
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
    shapelet_lengths = [360, 180, 90, 45]
    forecast_lags = [0, 5, 10, 60]
    shapelet_options: list[int | str] = ["all", 100, 50, 25, 10, 5]

    results_df = run_experiments(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=dataset.classes_,
        shapelet_lengths=shapelet_lengths,
        forecast_lags=forecast_lags,
        shapelet_options=shapelet_options,
        event_index=event_onset_index,
        observation_window_size=observation_window_size,
        random_state=42,
    )

    output_dir = data_root / "reports" / "experiment11"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"experiment11_results_{stamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    val_heatmap_path = output_dir / f"experiment11_val_metrics_heatmap_{stamp}.png"
    plot_metric_heatmaps(results_df, val_heatmap_path, split_prefix="val")
    print(f"Saved validation plot: {val_heatmap_path}")

    test_heatmap_path = output_dir / f"experiment11_test_metrics_heatmap_{stamp}.png"
    plot_metric_heatmaps(results_df, test_heatmap_path, split_prefix="test")
    print(f"Saved test plot: {test_heatmap_path}")

    best = select_best(results_df)
    best_path = output_dir / f"experiment11_best_config_{stamp}.csv"
    best.to_frame().T.to_csv(best_path, index=False)

    print("\n" + "=" * 70)
    print("Best configuration (sorted by val_tss, val_hss, val_f1, then test ties):")
    for col in [
        "window_size",
        "forecast_lag_min",
        "coeff_setting",
        "max_available_coeffs",
        "effective_coeffs",
        "val_tss",
        "val_hss",
        "val_css",
        "val_accuracy",
        "val_f1",
        "val_precision",
        "val_recall",
        "test_tss",
        "test_hss",
        "test_css",
        "test_accuracy",
        "test_f1",
        "test_precision",
        "test_recall",
    ]:
        print(f"{col}: {best[col]}")
    print("=" * 70)
    print(f"Saved best config: {best_path}")


if __name__ == "__main__":
    main()
