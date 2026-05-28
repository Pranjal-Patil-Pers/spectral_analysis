"""
Experiment 3 (SWAN-SF): Bulk counterfactual evaluation on the best model.

Loads the best model from the most recent experiment2 run and generates
counterfactuals for 100 instances from each class (FL and NF), then computes
per-instance, per-class, and overall DiCE quality metrics.

Data format: 60 timesteps per file, 12-minute cadence → 720-minute (12-hour) window.
All index/lag/window parameters are in TIMESTEPS (1 timestep = 12 minutes).

  EVENT_INDEX         = 60   rows = 720 min = 12 h
  OBSERVATION_WINDOW  = 48   rows = 576 min = 9.6 h

Outputs (under data/swansf/reports/experiment3/<run_stamp>/):
  - swansf_exp3_bulk_cf_instances_<stamp>.csv   : one row per attempted CF
  - swansf_exp3_bulk_cf_per_class_<stamp>.csv   : metric summary by source class
  - swansf_exp3_bulk_cf_overall_<stamp>.csv     : aggregate metrics across all CFs
  - swansf_exp3_bulk_cf_gallery_<stamp>.png     : gallery of up to 12 sampled CFs
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment15 import (
    build_fft_feature_names,
    build_fft_features_all,
    extract_observation_window,
    percent_key,
)
from experiment16 import (
    build_best_model_feature_bank,
    choose_training_examples_for_counterfactuals,
    make_dice_explainer,
    reconstruct_observation_from_fft_features,
)
from experiment2 import (
    FEATURE_COLS,
    SWANSFDataset,
    _compute_cf_metrics,
    _compute_diversity_metrics,
    _fit_dice_metric_stats,
    _plot_cf_timeseries,
    get_swansf_splits_with_ids,
    make_inverse_minmax,
    plot_counterfactual_gallery,
    summarize_counterfactual_quality,
)

_NUM_PER_CLASS = 100


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _find_latest_exp2_best_artifact(
    swansf_dir: Path,
) -> tuple[dict, pd.Series, str, Path]:
    """Return (artifact, best_row, run_stamp, exp2_run_dir) for the latest exp2 run."""
    exp2_root = swansf_dir / "reports" / "experiment2"
    run_dirs = sorted(
        [d for d in exp2_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No experiment2 run directories found under {exp2_root}. "
            "Run experiment2.py first."
        )

    for run_dir in run_dirs:
        summary_files = sorted(run_dir.glob("swansf_exp2_best_model_summary_*.csv"))
        if not summary_files:
            continue
        summary_path = summary_files[-1]
        run_stamp = run_dir.name

        best_row = pd.read_csv(summary_path).iloc[0]
        model_path = (
            run_dir
            / "saved_models"
            / f"experiment16_rf_w{int(best_row['window_size'])}"
            f"_lag{int(best_row['forecast_lag_min'])}"
            f"_top{int(percent_key(float(best_row['top_percent'])))}_{run_stamp}.pkl"
        )
        if not model_path.exists():
            continue

        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        return artifact, best_row, run_stamp, run_dir

    raise FileNotFoundError(
        "Could not find a valid experiment2 best-model artifact. "
        "Ensure experiment2 completed successfully."
    )


# ---------------------------------------------------------------------------
# Bulk CF generation
# ---------------------------------------------------------------------------

def run_bulk_counterfactual_evaluation(
    artifact: dict,
    X_train: list[np.ndarray],
    y_train: np.ndarray,
    X_eval: list[np.ndarray],
    y_eval: np.ndarray,
    eval_ids: list[str],
    classes: np.ndarray,
    channel_names: list[str],
    event_index: int,
    observation_window_size: int,
    output_dir: Path,
    run_stamp: str,
    inverse_transform_fn: callable,
    num_per_class: int = _NUM_PER_CLASS,
    num_cfs_per_instance: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate counterfactuals for up to num_per_class instances from each class.

    Individual timeseries plots are skipped; a sampled gallery is saved instead.

    Returns
    -------
    per_instance_df  : one row per attempted CF with full metrics
    per_class_df     : metric means/medians broken down by source class
    overall_df       : aggregate summary across all instances
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_bank = build_best_model_feature_bank(artifact, X_train, event_index, observation_window_size)
    eval_bank  = build_best_model_feature_bank(artifact, X_eval,  event_index, observation_window_size)

    X_train_flat  = np.asarray(train_bank["X_train_flat"], dtype=np.float32)
    X_eval_flat   = np.asarray(eval_bank["X_train_flat"],  dtype=np.float32)
    sel_idx       = np.asarray(artifact["selected_feature_indices"], dtype=np.int64)
    X_train_sel   = X_train_flat[:, sel_idx]
    X_eval_sel    = X_eval_flat[:, sel_idx]
    feature_names = list(artifact["selected_feature_names"])
    model         = artifact["model"]

    channels   = int(train_bank["channels"])
    n_slices   = int(train_bank["n_slices"])
    max_coeffs = int(train_bank["max_coeffs"])
    fft_window = int(artifact["window_size"])
    lag        = int(artifact["forecast_lag_min"])

    metric_stats = _fit_dice_metric_stats(X_train_sel)
    dice = make_dice_explainer(X_train_sel, y_train, feature_names, model)
    permitted_range = {name: [0.0, 1.0] for name in feature_names}

    chosen = choose_training_examples_for_counterfactuals(
        X_examples_sel=X_eval_sel,
        y_examples=y_eval,
        example_ids=eval_ids,
        model=model,
        classes=classes,
        num_per_class=num_per_class,
    )
    print(
        f"  Selected {len(chosen)} instances "
        f"({num_per_class} per class requested, {len(classes)} classes)"
    )

    _avg_keys = {
        "changed_feature_count", "changed_feature_fraction",
        "dice_proximity_loss", "dice_proximity_score",
        "dice_sparsity_loss", "dice_sparsity_score",
        "dice_plausibility_distance", "dice_plausibility_score",
    }
    empty_metrics: dict = {
        "changed_feature_count": 0,
        "selected_feature_count": len(feature_names),
        "changed_feature_fraction": np.nan,
        "dice_proximity_loss": np.nan,    "dice_proximity_score": np.nan,
        "dice_sparsity_loss": np.nan,     "dice_sparsity_score": np.nan,
        "dice_plausibility_distance": np.nan, "dice_plausibility_score": np.nan,
        "dice_nearest_train_index": -1,   "dice_nearest_train_label": -1,
    }
    empty_diversity: dict = {
        "dice_diversity_dpp": np.nan, "dice_diversity_avg_dist": np.nan,
        "dice_diversity_pair_count": 0, "dice_mean_pairwise_distance": np.nan,
    }

    rows: list[dict] = []
    plot_records: list[dict] = []
    cf_vectors_by_class: dict[int, list[np.ndarray]] = {}

    for n, ex in enumerate(chosen):
        if n == 0 or (n + 1) % 10 == 0 or (n + 1) == len(chosen):
            print(f"    [{n + 1:3d}/{len(chosen)}] {ex['label_name']:3s} → {ex['sample_id']}")

        idx               = int(ex["example_index"])
        sample_id         = str(ex["sample_id"])
        label             = int(ex["label"])
        label_name        = str(ex["label_name"])
        target_label      = 1 - label
        target_label_name = str(classes[target_label])

        original_full = X_eval_flat[idx].astype(np.float64).copy()
        original_sel  = X_eval_sel[idx].astype(np.float64).copy()
        query_df      = pd.DataFrame([original_sel], columns=feature_names)

        cf_generated   = False
        cf_prediction  = int(ex["predicted_label"])
        error_msg      = ""
        cf_mse         = np.nan
        cf_mae         = np.nan
        cf_metrics     = dict(empty_metrics)
        inst_diversity = dict(empty_diversity)
        num_cfs_generated = 0

        try:
            cf_result = dice.generate_counterfactuals(
                query_df, total_CFs=num_cfs_per_instance, desired_class="opposite",
                features_to_vary="all", permitted_range=permitted_range,
                verbose=False, sample_size=2000, random_seed=random_state + idx,
            )
            cf_df_rows = cf_result.cf_examples_list[0].final_cfs_df
            if len(cf_df_rows) == 0:
                raise RuntimeError("DiCE returned no rows.")

            original_obs = extract_observation_window(
                X_eval[idx], event_index=event_index,
                observation_window_size=observation_window_size, lag_minute=lag,
            )

            inst_cf_sels: list[np.ndarray] = []
            inst_cf_metrics_list: list[dict] = []
            inst_cf_mse_list: list[float] = []
            inst_cf_mae_list: list[float] = []

            for i in range(len(cf_df_rows)):
                cf_sel_i = cf_df_rows.iloc[i][feature_names].astype(float).to_numpy(dtype=np.float64)
                cf_full_i = original_full.copy()
                cf_full_i[sel_idx] = cf_sel_i
                cf_recon_i = reconstruct_observation_from_fft_features(
                    cf_full_i, channels=channels, n_slices=n_slices,
                    max_coeffs=max_coeffs, fft_window_size=fft_window,
                )
                cf_pred_i = int(model.predict(cf_sel_i.reshape(1, -1))[0])
                cf_metrics_i = _compute_cf_metrics(
                    original_sel=original_sel, cf_selected=cf_sel_i,
                    X_train_sel=X_train_sel, y_train=y_train,
                    cf_prediction=cf_pred_i, metric_stats=metric_stats,
                )
                inst_cf_sels.append(cf_sel_i)
                inst_cf_metrics_list.append(cf_metrics_i)
                inst_cf_mse_list.append(float(np.mean((original_obs - cf_recon_i) ** 2)))
                inst_cf_mae_list.append(float(np.mean(np.abs(original_obs - cf_recon_i))))

            num_cfs_generated = len(inst_cf_sels)
            best_i = int(np.argmin([m["dice_proximity_loss"] for m in inst_cf_metrics_list]))
            cf_prediction = int(model.predict(inst_cf_sels[best_i].reshape(1, -1))[0])

            cf_metrics = {
                k: float(np.mean([m[k] for m in inst_cf_metrics_list])) if k in _avg_keys
                   else inst_cf_metrics_list[best_i][k]
                for k in inst_cf_metrics_list[0]
            }
            cf_mse = float(np.mean(inst_cf_mse_list))
            cf_mae = float(np.mean(inst_cf_mae_list))
            inst_diversity = _compute_diversity_metrics(inst_cf_sels, metric_stats)

            cf_vectors_by_class.setdefault(label, []).extend(inst_cf_sels)

            cf_full_best = original_full.copy()
            cf_full_best[sel_idx] = inst_cf_sels[best_i]
            cf_recon_best = reconstruct_observation_from_fft_features(
                cf_full_best, channels=channels, n_slices=n_slices,
                max_coeffs=max_coeffs, fft_window_size=fft_window,
            )
            best_mse = float(np.mean((original_obs - cf_recon_best) ** 2))
            plot_records.append({
                "sample_id": sample_id,
                "label_name": label_name,
                "target_label_name": target_label_name,
                "original_obs": original_obs,
                "cf_recon": cf_recon_best,
                "lag": lag,
                "window_size": fft_window,
                "top_percent_key": int(artifact["top_percent_key"]),
                "counterfactual_mse": best_mse,
                "channel_names": channel_names,
            })
            cf_generated = True

        except Exception as exc:
            error_msg = str(exc)

        rows.append({
            "sample_id": sample_id,
            "original_label": label,
            "original_label_name": label_name,
            "desired_counterfactual_label": target_label,
            "desired_counterfactual_label_name": target_label_name,
            "counterfactual_predicted_label": cf_prediction,
            "counterfactual_predicted_label_name": str(classes[cf_prediction]),
            "counterfactual_found": bool(cf_generated),
            "num_cfs_requested": num_cfs_per_instance,
            "num_cfs_generated": num_cfs_generated,
            "counterfactual_reconstruction_mse": float(cf_mse) if np.isfinite(cf_mse) else np.nan,
            "counterfactual_reconstruction_mae": float(cf_mae) if np.isfinite(cf_mae) else np.nan,
            **cf_metrics,
            **inst_diversity,
            "forecast_lag_min": lag,
            "window_size": fft_window,
            "top_percent": float(artifact["top_percent"]),
            "error": error_msg,
        })

    per_instance_df = pd.DataFrame(rows)

    # --- per-class metric summary
    metric_cols = [
        "dice_proximity_loss",   "dice_proximity_score",
        "dice_sparsity_loss",    "dice_sparsity_score",
        "dice_plausibility_distance", "dice_plausibility_score",
        "changed_feature_count", "changed_feature_fraction",
        "counterfactual_reconstruction_mse", "counterfactual_reconstruction_mae",
        "dice_diversity_dpp", "dice_diversity_avg_dist",
        "dice_mean_pairwise_distance", "num_cfs_generated",
    ]
    class_rows: list[dict] = []
    for cls_idx in sorted(per_instance_df["original_label"].unique()):
        cls_df  = per_instance_df[per_instance_df["original_label"] == int(cls_idx)]
        found   = cls_df[cls_df["counterfactual_found"].astype(bool)]
        div     = _compute_diversity_metrics(
            cf_vectors_by_class.get(int(cls_idx), []), metric_stats
        )
        class_row: dict = {
            "original_label":      int(cls_idx),
            "original_label_name": str(classes[int(cls_idx)]),
            "attempted":           int(len(cls_df)),
            "found":               int(len(found)),
            "success_rate":        float(len(found) / len(cls_df)) if len(cls_df) > 0 else np.nan,
        }
        for col in metric_cols:
            vals = pd.to_numeric(found[col], errors="coerce") if col in found else pd.Series(dtype=float)
            class_row[f"mean_{col}"]   = float(vals.mean())   if len(vals.dropna()) > 0 else np.nan
            class_row[f"median_{col}"] = float(vals.median()) if len(vals.dropna()) > 0 else np.nan
        class_row.update(div)
        class_rows.append(class_row)
    per_class_df = pd.DataFrame(class_rows)

    # --- overall summary (diversity aggregated from per-instance values)
    overall_df = summarize_counterfactual_quality(per_instance_df)

    # --- gallery: up to 6 CFs per class (12 total)
    gallery_records: list[dict] = []
    for cls_name in [str(c) for c in classes]:
        cls_plots = [r for r in plot_records if r["label_name"] == cls_name]
        gallery_records.extend(cls_plots[:6])
    plot_counterfactual_gallery(
        gallery_records,
        output_dir / f"swansf_exp3_bulk_cf_gallery_{run_stamp}.png",
        title=(
            f"SWAN-SF Bulk CF Gallery — "
            f"{len(plot_records)} found / {len(rows)} attempted"
        ),
    )

    return per_instance_df, per_class_df, overall_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    swansf_dir   = project_root / "data" / "swansf"

    # --- locate experiment1 output
    stride_dirs = sorted(
        swansf_dir.glob("partition4_stride_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not stride_dirs:
        raise FileNotFoundError(
            f"No partition4_stride_* directory found under {swansf_dir}. "
            "Run experiment1.py first."
        )
    data_dir        = stride_dirs[0]
    manifest_path   = data_dir / "selected_files_manifest.csv"
    sample_csv_dir  = data_dir / "sample_csv_files"
    norm_meta_path  = data_dir / "normalization_metadata.json"

    print(f"Data directory    : {data_dir}")

    with open(norm_meta_path, encoding="utf-8") as f:
        norm_meta = json.load(f)
    inverse_transform = make_inverse_minmax(norm_meta["stats"], FEATURE_COLS)

    dataset = SWANSFDataset(
        manifest_path=manifest_path,
        sample_csv_dir=sample_csv_dir,
        feature_cols=FEATURE_COLS,
    )
    (X_train, y_train, _), (X_val, y_val, _), (X_test, y_test, test_ids) = (
        get_swansf_splits_with_ids(dataset)
    )

    # --- load best artifact from latest experiment2 run
    print("Loading best model from experiment2...")
    artifact, best_row, exp2_stamp, exp2_run_dir = _find_latest_exp2_best_artifact(swansf_dir)
    lag_ts  = int(best_row["forecast_lag_min"])
    win_ts  = int(best_row["window_size"])
    print(
        f"  Source run        : {exp2_stamp}\n"
        f"  lag={lag_ts} rows ({lag_ts * 12} min), "
        f"window={win_ts} rows ({win_ts * 12} min), "
        f"top%={int(round(float(best_row['top_percent']) * 100))}, "
        f"val_css={float(best_row['val_css']):.4f}, "
        f"test_css={float(best_row['test_css']):.4f}"
    )

    # same parameters as experiment2 (in timesteps; 1 timestep = 12 min)
    EVENT_INDEX        = 60  # 720 min = 12 h
    OBSERVATION_WINDOW = 48  # 576 min = 9.6 h

    run_stamp  = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = swansf_dir / "reports" / "experiment3" / run_stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory  : {output_dir}\n")

    # --- run bulk evaluation
    print(f"Generating {_NUM_PER_CLASS} CFs per class...")
    per_instance_df, per_class_df, overall_df = run_bulk_counterfactual_evaluation(
        artifact=artifact,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        eval_ids=test_ids,
        classes=dataset.classes_,
        channel_names=FEATURE_COLS,
        event_index=EVENT_INDEX,
        observation_window_size=OBSERVATION_WINDOW,
        output_dir=output_dir,
        run_stamp=run_stamp,
        inverse_transform_fn=inverse_transform,
        num_per_class=_NUM_PER_CLASS,
        num_cfs_per_instance=5,
        random_state=42,
    )

    # --- save outputs
    instances_path  = output_dir / f"swansf_exp3_bulk_cf_instances_{run_stamp}.csv"
    per_class_path  = output_dir / f"swansf_exp3_bulk_cf_per_class_{run_stamp}.csv"
    overall_path    = output_dir / f"swansf_exp3_bulk_cf_overall_{run_stamp}.csv"
    gallery_path    = output_dir / f"swansf_exp3_bulk_cf_gallery_{run_stamp}.png"

    per_instance_df.to_csv(instances_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    # --- print summary
    print("\n" + "=" * 80)
    print("Experiment 3 (SWAN-SF bulk CFs) finished.")
    print(f"  Instance-level CSV  : {instances_path}")
    print(f"  Per-class CSV       : {per_class_path}")
    print(f"  Overall CSV         : {overall_path}")
    print(f"  Gallery             : {gallery_path}")

    print("\n  Per-class results:")
    for _, row in per_class_df.iterrows():
        print(
            f"    {row['original_label_name']:3s} → opposite | "
            f"attempted={int(row['attempted'])}, "
            f"found={int(row['found'])}, "
            f"success={float(row['success_rate']):.2%}, "
            f"proximity={float(row.get('mean_dice_proximity_score', np.nan)):.4f}, "
            f"sparsity={float(row.get('mean_dice_sparsity_score', np.nan)):.4f}, "
            f"plausibility={float(row.get('mean_dice_plausibility_score', np.nan)):.4f}"
        )

    if len(overall_df) > 0:
        o = overall_df.iloc[0]
        print(
            f"\n  Overall | "
            f"success={float(o.get('counterfactual_success_rate', np.nan)):.2%}, "
            f"proximity={float(o.get('mean_dice_proximity_score', np.nan)):.4f}, "
            f"sparsity={float(o.get('mean_dice_sparsity_score', np.nan)):.4f}, "
            f"plausibility={float(o.get('mean_dice_plausibility_score', np.nan)):.4f}, "
            f"diversity_dpp={float(o.get('dice_diversity_dpp', np.nan)):.4e}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
