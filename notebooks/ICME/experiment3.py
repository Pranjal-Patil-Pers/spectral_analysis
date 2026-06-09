"""
ICME Experiment 3 — Large-scale counterfactual evaluation.

Generates 100 counterfactuals for each of 100 instances (50 per class),
then computes proximity, sparsity, plausibility, and diversity for every instance.

Pipeline is identical to Experiment 2 (Box-Cox → FFT mag+phase → RF).
The key difference is scale: 100 CFs per instance so that per-instance
diversity metrics are statistically meaningful.

Output: data/ICMECAT/reports/experiment3/
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import dice_ml

# reuse all helpers from experiment2
from experiment2 import (
    apply_box_cox,
    build_feature_matrix,
    compute_cf_metrics,
    compute_diversity_metrics,
    fit_box_cox,
    fit_dice_metric_stats,
    invert_box_cox,
    load_icme_dataset,
    plot_cf_gallery,
    plot_cf_timeseries,
    reconstruct_from_fft_features,
    resample_series,
    summarize_metrics,
    _SelectedFeatureRF,
)

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "ICMECAT"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "ICMECAT" / "reports" / "experiment3"

SERIES_LEN             = 48
N_TREES                = 300
RANDOM_STATE           = 42
NUM_INSTANCES_PER_CLASS = 50   # 50 quiet + 50 ICME = 100 total
NUM_CF                 = 5   # counterfactuals generated per instance


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    run_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cf_ts_dir = OUTPUT_DIR / "counterfactual_timeseries" / run_stamp
    cf_ts_dir.mkdir(parents=True, exist_ok=True)

    classes = np.array(["quiet", "icme"])

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading ICME dataset …")
    X_raw, y, ids = load_icme_dataset(DATA_DIR)
    print(f"  {len(X_raw)} events — ICME: {(y==1).sum()}, quiet: {(y==0).sum()}")

    # ── 2. Stratified 70 / 15 / 15 split ─────────────────────────────────────
    idx = np.arange(len(X_raw))
    idx_tv, idx_test, y_tv, y_test = train_test_split(
        idx, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_tv, y_tv, test_size=0.15 / 0.85, stratify=y_tv, random_state=RANDOM_STATE
    )
    X_train_raw = [X_raw[i] for i in idx_train]
    X_val_raw   = [X_raw[i] for i in idx_val]
    X_test_raw  = [X_raw[i] for i in idx_test]
    ids_test    = [ids[i] for i in idx_test]
    print(f"  Train {len(X_train_raw)} / Val {len(X_val_raw)} / Test {len(X_test_raw)}")

    # ── 3. Box-Cox fit ────────────────────────────────────────────────────────
    print("Fitting Box-Cox transform …")
    shift, lam = fit_box_cox(X_train_raw)
    print(f"  shift={shift:.6f}  lambda={lam:.4f}")

    # ── 4. FFT feature matrices ───────────────────────────────────────────────
    print(f"Extracting FFT features (magnitude + phase, series_len={SERIES_LEN}) …")
    X_train_feat, feature_names = build_feature_matrix(X_train_raw, SERIES_LEN, shift, lam)
    X_val_feat,   _             = build_feature_matrix(X_val_raw,   SERIES_LEN, shift, lam)
    X_test_feat,  _             = build_feature_matrix(X_test_raw,  SERIES_LEN, shift, lam)
    print(f"  Shape: {X_train_feat.shape}  ({len(feature_names)} features)")

    # ── 5. Train Random Forest ────────────────────────────────────────────────
    print("Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_feat, y_train)
    val_acc  = rf.score(X_val_feat,  y_val)
    test_acc = rf.score(X_test_feat, y_test)
    print(f"  Val accuracy : {val_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(classification_report(y_test, rf.predict(X_test_feat), target_names=["quiet", "icme"]))

    # ── 6. Top-50% cumulative-importance feature selection ────────────────────
    importances  = rf.feature_importances_
    sorted_idx   = np.argsort(importances)[::-1]
    cumsum       = np.cumsum(importances[sorted_idx])
    n_top        = int(np.searchsorted(cumsum, 0.50)) + 1
    selected_idx   = sorted_idx[:n_top]
    selected_names = [feature_names[i] for i in selected_idx]
    n_mag   = sum(1 for n in selected_names if n.startswith("mag_"))
    n_phase = sum(1 for n in selected_names if n.startswith("phase_"))
    print(f"  Top-50% features: {n_top}/{len(feature_names)}  (magnitude: {n_mag}, phase: {n_phase})")

    X_train_sel = X_train_feat[:, selected_idx]
    X_test_sel  = X_test_feat[:,  selected_idx]

    # ── 7. Save model ─────────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / f"icme3_rf_{run_stamp}.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({
            "model":                    rf,
            "feature_names":            feature_names,
            "selected_feature_names":   selected_names,
            "selected_feature_indices": selected_idx.tolist(),
            "shift": shift, "lambda_": lam, "series_len": SERIES_LEN,
            "val_accuracy": val_acc, "test_accuracy": test_acc,
        }, fh)
    print(f"  Model saved → {model_path}")

    # ── 8. DiCE setup ─────────────────────────────────────────────────────────
    print("Setting up DiCE …")
    metric_stats = fit_dice_metric_stats(X_train_sel)

    train_df           = pd.DataFrame(X_train_sel, columns=selected_names)
    train_df["label"]  = y_train.astype(int)
    wrapped_model      = _SelectedFeatureRF(rf, selected_idx)
    dice_data          = dice_ml.Data(
        dataframe=train_df, continuous_features=selected_names, outcome_name="label"
    )
    dice_model = dice_ml.Model(model=wrapped_model, backend="sklearn")
    exp_dice   = dice_ml.Dice(dice_data, dice_model, method="random")

    # ── 9. Select 50 instances per class from test set ────────────────────────
    # Prefer correctly classified; fall back to any if not enough.
    def pick_instances(label: int) -> np.ndarray:
        mask    = y_test == label
        cand    = np.where(mask)[0]
        correct = cand[rf.predict(X_test_feat[cand]) == label]
        pool    = correct if len(correct) >= NUM_INSTANCES_PER_CLASS else cand
        chosen  = pool[:NUM_INSTANCES_PER_CLASS]
        if len(chosen) < NUM_INSTANCES_PER_CLASS:
            print(f"  Warning: only {len(chosen)} instances available for class '{classes[label]}'")
        return chosen

    chosen_quiet = pick_instances(0)
    chosen_icme  = pick_instances(1)
    print(f"  Instances selected — quiet: {len(chosen_quiet)}, icme: {len(chosen_icme)}")

    # ── 10. Generate 100 CFs per instance ────────────────────────────────────
    print(f"Generating {NUM_CF} counterfactuals × "
          f"{len(chosen_quiet)+len(chosen_icme)} instances …")

    cf_rows: list[dict]      = []
    plot_records: list[dict] = []
    total_instances          = len(chosen_quiet) + len(chosen_icme)
    done                     = 0

    for orig_label, chosen in [(0, chosen_quiet), (1, chosen_icme)]:
        target_label      = 1 - orig_label
        label_name        = str(classes[orig_label])
        target_label_name = str(classes[target_label])

        for local_i in chosen:
            sample_id     = ids_test[local_i]
            query_df      = pd.DataFrame([X_test_sel[local_i]], columns=selected_names)
            cf_generated  = False
            mse           = np.nan
            error         = ""
            best_metrics: dict[str, float | int]      = {}
            diversity_inst: dict[str, float | int]    = compute_diversity_metrics([], metric_stats)

            try:
                cf_result = exp_dice.generate_counterfactuals(
                    query_df,
                    total_CFs=NUM_CF,
                    desired_class=int(target_label),
                    features_to_vary=selected_names,
                    verbose=False,
                    sample_size=500 * NUM_CF,   # scale sample budget with CF count
                    random_seed=RANDOM_STATE + int(local_i),
                )
                cf_df = cf_result.cf_examples_list[0].final_cfs_df
                if len(cf_df) == 0:
                    raise RuntimeError("DiCE returned no counterfactual rows.")

                # evaluate every returned CF
                instance_cf_vectors: list[np.ndarray] = []
                per_cf_metrics:      list[dict]       = []

                for _, cf_row in cf_df.iterrows():
                    cf_vals     = cf_row[selected_names].astype(float).to_numpy(dtype=np.float64)
                    cf_full_tmp = X_test_feat[local_i].copy()
                    cf_full_tmp[selected_idx] = cf_vals
                    cf_pred = int(rf.predict(cf_full_tmp.reshape(1, -1))[0])
                    m = compute_cf_metrics(
                        original_sel=X_test_sel[local_i],
                        cf_sel=cf_vals,
                        X_train_sel=X_train_sel,
                        y_train=y_train,
                        cf_predicted_label=cf_pred,
                        metric_stats=metric_stats,
                    )
                    instance_cf_vectors.append(cf_vals)
                    per_cf_metrics.append(m)

                # best CF = lowest proximity loss
                best_idx     = int(np.argmin([m["dice_proximity_loss"] for m in per_cf_metrics]))
                best_cf_vals = instance_cf_vectors[best_idx]
                best_metrics = per_cf_metrics[best_idx]

                # per-instance diversity over all returned CFs
                diversity_inst = compute_diversity_metrics(instance_cf_vectors, metric_stats)

                # reconstruct best CF → km/s
                cf_full = X_test_feat[local_i].copy()
                cf_full[selected_idx] = best_cf_vals
                original_resampled = resample_series(X_test_raw[local_i], SERIES_LEN)
                cf_recon = reconstruct_from_fft_features(cf_full, SERIES_LEN, shift, lam)
                mse = float(np.mean((original_resampled - cf_recon) ** 2))

                cf_generated = True
                plot_records.append({
                    "sample_id":         sample_id,
                    "original":          original_resampled,
                    "cf_recon":          cf_recon,
                    "label_name":        label_name,
                    "target_label_name": target_label_name,
                    "mse":               mse,
                })
                plot_cf_timeseries(
                    sample_id, original_resampled, cf_recon,
                    label_name, target_label_name, mse,
                    cf_ts_dir / f"cf_{sample_id}_{label_name}_to_{target_label_name}.png",
                )

            except Exception as exc:
                error = str(exc)

            cf_rows.append({
                "sample_id":            sample_id,
                "original_label":       orig_label,
                "original_label_name":  label_name,
                "target_label":         target_label,
                "target_label_name":    target_label_name,
                "counterfactual_found": cf_generated,
                "num_cf_requested":     NUM_CF,
                "num_cf_returned":      len(instance_cf_vectors) if cf_generated else 0,
                "reconstruction_mse":   mse,
                # proximity / sparsity / plausibility — on best CF
                "changed_feature_count":      best_metrics.get("changed_feature_count",      np.nan),
                "changed_feature_fraction":   best_metrics.get("changed_feature_fraction",   np.nan),
                "dice_proximity_loss":        best_metrics.get("dice_proximity_loss",        np.nan),
                "dice_proximity_score":       best_metrics.get("dice_proximity_score",       np.nan),
                "dice_sparsity_loss":         best_metrics.get("dice_sparsity_loss",         np.nan),
                "dice_sparsity_score":        best_metrics.get("dice_sparsity_score",        np.nan),
                "dice_plausibility_distance": best_metrics.get("dice_plausibility_distance", np.nan),
                "dice_plausibility_score":    best_metrics.get("dice_plausibility_score",    np.nan),
                # per-instance diversity over all 100 CFs
                **diversity_inst,
                "error": error,
            })

            done += 1
            print(f"  [{done:>3}/{total_instances}] {sample_id}  "
                  f"found={cf_generated}  "
                  f"n_cf={cf_rows[-1]['num_cf_returned']}", flush=True)

    # ── 11. Save results ──────────────────────────────────────────────────────
    cf_summary   = pd.DataFrame(cf_rows)
    summary_path = OUTPUT_DIR / f"icme3_counterfactual_summary_{run_stamp}.csv"
    cf_summary.to_csv(summary_path, index=False)

    # overall aggregated metrics
    metrics_df   = summarize_metrics(cf_summary)
    metrics_path = OUTPUT_DIR / f"icme3_counterfactual_metrics_{run_stamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # per-class aggregated metrics
    metric_cols = [
        "dice_proximity_loss",   "dice_proximity_score",
        "dice_sparsity_loss",    "dice_sparsity_score",
        "dice_plausibility_distance", "dice_plausibility_score",
        "dice_diversity_dpp",    "dice_diversity_avg_dist",
        "dice_diversity_pair_count", "dice_mean_pairwise_distance",
        "reconstruction_mse",    "changed_feature_count", "changed_feature_fraction",
    ]
    found_df = cf_summary[cf_summary["counterfactual_found"].astype(bool)]
    class_rows: list[dict] = []
    for lname in ["quiet", "icme", "all"]:
        sub = found_df if lname == "all" else found_df[found_df["original_label_name"] == lname]
        row: dict = {
            "class":                  lname,
            "n_instances":            len(sub),
            "counterfactual_success": len(sub),
        }
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna() if col in sub else pd.Series(dtype=float)
            row[f"mean_{col}"]   = float(vals.mean())   if len(vals) > 0 else float("nan")
            row[f"median_{col}"] = float(vals.median()) if len(vals) > 0 else float("nan")
            row[f"std_{col}"]    = float(vals.std())    if len(vals) > 0 else float("nan")
        class_rows.append(row)

    class_metrics_df   = pd.DataFrame(class_rows)
    class_metrics_path = OUTPUT_DIR / f"icme3_metrics_by_class_{run_stamp}.csv"
    class_metrics_df.to_csv(class_metrics_path, index=False)

    gallery_path = OUTPUT_DIR / f"icme3_counterfactual_gallery_{run_stamp}.png"
    plot_cf_gallery(plot_records, gallery_path, title="ICME Experiment 3 — Counterfactual Gallery")

    # ── 12. Print summary ─────────────────────────────────────────────────────
    found = int(cf_summary["counterfactual_found"].sum())
    total = len(cf_summary)
    print(f"\nCounterfactuals found : {found}/{total}  ({100*found/total:.1f}%)")

    if found > 0:
        print("\n── Aggregated metrics (mean ± std) ──────────────────────────────────")
        report_cols = [
            ("Proximity  score",   "dice_proximity_score"),
            ("Sparsity   score",   "dice_sparsity_score"),
            ("Plausibility score", "dice_plausibility_score"),
            ("Diversity avg dist", "dice_diversity_avg_dist"),
            ("Diversity DPP",      "dice_diversity_dpp"),
            ("Reconstruction MSE", "reconstruction_mse"),
        ]
        for label, col in report_cols:
            vals = pd.to_numeric(found_df[col], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"  {label:<22}  mean={vals.mean():.4f}  "
                      f"std={vals.std():.4f}  median={vals.median():.4f}")

        print("\n── By original class ────────────────────────────────────────────────")
        for row in class_rows[:2]:   # quiet and icme rows
            print(f"  {row['class']:<6}  n={row['n_instances']:<3}  "
                  f"prox={row['mean_dice_proximity_score']:.3f}  "
                  f"spar={row['mean_dice_sparsity_score']:.3f}  "
                  f"plau={row['mean_dice_plausibility_score']:.3f}  "
                  f"div={row['mean_dice_diversity_avg_dist']:.3f}")

    print(f"\nSummary CSV       → {summary_path}")
    print(f"Overall metrics   → {metrics_path}")
    print(f"Per-class metrics → {class_metrics_path}")
    print(f"Gallery           → {gallery_path}")
    print("\n" + "=" * 70)
    print("Experiment ICME3 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
