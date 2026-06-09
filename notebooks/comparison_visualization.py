"""
Cross-dataset comparison visualization: SWANSF, ICME Solar Winds, SEP counterfactual metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Data ──────────────────────────────────────────────────────────────────────

data = {
    "Dataset": ["SWANSF", "ICME Solar Winds", "SEP"],
    "generated_counterfactual_count":         [200,  100,  100],
    "attempted_counterfactual_count":          [200,  100,  100],
    "counterfactual_success_rate":             [1.0,  1.0,  1.0],
    "mean_dice_proximity_score":               [0.274533356, 0.857733204, 0.469301075],
    "mean_dice_sparsity_score":                [0.756880952, 0.921578947, 0.837054795],
    "mean_dice_plausibility_score":            [0.174787776, 0.576862377, 0.260384885],
    "mean_changed_feature_count":              [40.844,  1.49,  23.79],
    "mean_changed_feature_fraction":           [0.243119048, 0.078421053, 0.162945205],
    "mean_counterfactual_reconstruction_mse":  [0.015915461, 15835.87825, 5971.182774],
    "median_counterfactual_reconstruction_mse":[0.010388266, 31.79723765, 0.07127177],
}

df = pd.DataFrame(data).set_index("Dataset")

COLORS = ["#4C72B0", "#DD8452", "#55A868"]
DATASETS = df.index.tolist()

# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 18))
fig.suptitle(
    "Cross-Dataset Counterfactual Explanation Metrics\n(SWANSF · ICME Solar Winds · SEP)",
    fontsize=16, fontweight="bold", y=0.98,
)

# Grid: 3 rows × 3 cols + 1 extra wide panel for reconstruction MSE
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.4)

ax_proximity   = fig.add_subplot(gs[0, 0])
ax_sparsity    = fig.add_subplot(gs[0, 1])
ax_plausibility= fig.add_subplot(gs[0, 2])
ax_feat_count  = fig.add_subplot(gs[1, 0])
ax_feat_frac   = fig.add_subplot(gs[1, 1])
ax_success     = fig.add_subplot(gs[1, 2])
ax_mse_mean    = fig.add_subplot(gs[2, 0:2])   # wide
ax_mse_median  = fig.add_subplot(gs[2, 2])


def bar_chart(ax, column, title, ylabel, color_list=COLORS, log=False):
    vals = df[column].values
    x = np.arange(len(DATASETS))
    bars = ax.bar(x, vals, color=color_list, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=9, rotation=15, ha="right")
    ax.yaxis.set_tick_params(labelsize=8)
    if log:
        ax.set_yscale("log")
    for bar, v in zip(bars, vals):
        label = f"{v:.3g}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * (1.04 if not log else 1.3),
            label, ha="center", va="bottom", fontsize=8,
        )
    ax.spines[["top", "right"]].set_visible(False)


# Row 1 – DiCE quality scores
bar_chart(ax_proximity,    "mean_dice_proximity_score",  "Mean DiCE Proximity Score",    "Score (higher = closer)")
bar_chart(ax_sparsity,     "mean_dice_sparsity_score",   "Mean DiCE Sparsity Score",     "Score (higher = sparser)")
bar_chart(ax_plausibility, "mean_dice_plausibility_score","Mean DiCE Plausibility Score","Score (higher = more plausible)")

# Row 2 – Feature change metrics + success rate
bar_chart(ax_feat_count, "mean_changed_feature_count",    "Mean Changed Feature Count",    "# Features changed")
bar_chart(ax_feat_frac,  "mean_changed_feature_fraction", "Mean Changed Feature Fraction", "Fraction of features")

# Success rate as a horizontal reference bar chart (all 1.0, so show as annotation)
ax_success.bar(np.arange(len(DATASETS)), [1.0] * 3, color=COLORS, edgecolor="white")
ax_success.set_title("Counterfactual Success Rate", fontsize=11, fontweight="bold", pad=8)
ax_success.set_ylabel("Rate", fontsize=9)
ax_success.set_ylim(0, 1.3)
ax_success.set_xticks(np.arange(len(DATASETS)))
ax_success.set_xticklabels(DATASETS, fontsize=9, rotation=15, ha="right")
for i in range(3):
    ax_success.text(i, 1.06, "100 %", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_success.spines[["top", "right"]].set_visible(False)

# Row 3 – Reconstruction MSE (log scale because of huge range)
bar_chart(ax_mse_mean,   "mean_counterfactual_reconstruction_mse",
          "Mean Reconstruction MSE  (log scale)", "MSE", log=True)
bar_chart(ax_mse_median, "median_counterfactual_reconstruction_mse",
          "Median Reconstruction MSE  (log scale)", "MSE", log=True)

# ── Legend ────────────────────────────────────────────────────────────────────

patches = [mpatches.Patch(color=c, label=d) for c, d in zip(COLORS, DATASETS)]
fig.legend(
    handles=patches, loc="lower center", ncol=3,
    fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01),
)

# ── Save ──────────────────────────────────────────────────────────────────────

out_path = "data/comparison_metrics.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()


# ── Radar / spider chart – DiCE quality scores ────────────────────────────────

radar_metrics = [
    "mean_dice_proximity_score",
    "mean_dice_sparsity_score",
    "mean_dice_plausibility_score",
    "mean_changed_feature_fraction",
]
radar_labels = ["Proximity", "Sparsity", "Plausibility", "Changed\nFraction"]

angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig2, ax_r = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig2.suptitle(
    "DiCE Quality Scores – Radar Comparison\n(values normalised to [0, 1])",
    fontsize=13, fontweight="bold",
)

for dataset, color in zip(DATASETS, COLORS):
    vals_raw = df.loc[dataset, radar_metrics].values.astype(float)
    # Normalise each metric across datasets so all axes are 0-1
    col_min = df[radar_metrics].min().values
    col_max = df[radar_metrics].max().values
    vals_norm = (vals_raw - col_min) / np.where(col_max - col_min == 0, 1, col_max - col_min)
    vals_norm = list(vals_norm) + [vals_norm[0]]
    ax_r.plot(angles, vals_norm, "o-", linewidth=2, color=color, label=dataset)
    ax_r.fill(angles, vals_norm, alpha=0.12, color=color)

ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels(radar_labels, fontsize=11)
ax_r.set_ylim(0, 1)
ax_r.set_yticks([0.25, 0.5, 0.75, 1.0])
ax_r.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
ax_r.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)

out_path2 = "data/comparison_radar.png"
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")
plt.show()
