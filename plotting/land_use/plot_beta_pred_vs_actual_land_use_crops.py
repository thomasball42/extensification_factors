"""
Plots how well beta predicts the land-use (area harvested) change that
actually happened, from the per-transition and per-series CSVs produced by
../../validation/compute_beta_pred_vs_actual_land_use_crops.py -- run that
script first. See its docstring for how the predicted area, prediction
error, and scenario classification (A/B/C, by size of the actual production
change) are derived.

Produces:
  land_use_pred_vs_actual_scatter.png           -- predicted vs. actual
                                                     log-change in area (all
                                                     transitions pooled),
                                                     unweighted vs.
                                                     production-weighted
  land_use_pred_vs_actual_error_distribution.png -- 3x2 grid: one row per
                                                     scenario (A/B/C), one
                                                     column per unweighted /
                                                     production-weighted,
                                                     distribution of the
                                                     prediction error (actual
                                                     - predicted, as % of
                                                     actual area)
  land_use_pred_vs_actual_by_item.png            -- top items BY CURRENT
                                                     PRODUCTION (not by
                                                     error), grouped bars of
                                                     production-weighted mean
                                                     |% error| per scenario
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _utils import load_plot_config, weighted_group_mean

SCRIPT_NAME = "plot_beta_pred_vs_actual_land_use_crops"

TRANSITIONS_PATH = Path("outputs") / "land_use_impact" / "land_use_pred_vs_actual_transitions_crops.csv"
SERIES_PATH = Path("outputs") / "land_use_impact" / "land_use_pred_vs_actual_scenarios_crops.csv"
figs_path = Path("..") / "figs" / "land_use" / "beta_pred_vs_actual_land_use_crops"
SAVE = True

_lui_cfg = load_plot_config()["land_use_impact"][SCRIPT_NAME]
TOP_N_ITEMS = _lui_cfg["top_n_items"]
MIN_SERIES_PER_ITEM = _lui_cfg.get("min_series_per_item", 3)  # min qualifying transitions per item, per scenario

BASE_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
COLOR_CORRECT = "#1baf7a"
COLOR_WRONG = "#e34948"
SCENARIO_COLORS = [BASE_PALETTE[1], BASE_PALETTE[6], BASE_PALETTE[3]]
SCENARIO_KEYS = ["a", "b", "c"]

os.makedirs(figs_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the per-transition and per-series CSVs written by
# compute_beta_pred_vs_actual_land_use_crops.py
# ---------------------------------------------------------------------------

trans = pd.read_csv(TRANSITIONS_PATH)
series = pd.read_csv(SERIES_PATH)
SCENARIO_LABELS = dict(trans[["scenario", "scenario_label"]].drop_duplicates().to_numpy())

n_series = trans[["Area", "Item", "Item Code"]].drop_duplicates().shape[0]
print(f"{len(trans)} transitions across {n_series} series loaded from {TRANSITIONS_PATH.name}.")

actual_a = trans["actual_a"].to_numpy(float)
predicted_a = trans["predicted_a"].to_numpy(float)
correct_direction = trans["correct_direction"].to_numpy(bool)
weight = trans["Production"].to_numpy(float)

r2 = np.corrcoef(actual_a, predicted_a)[0, 1] ** 2
dir_acc = correct_direction.mean()
dir_acc_w = np.average(correct_direction, weights=weight)
print(f"R-squared (predicted vs. actual log-change in area): {r2:.3f}")
print(f"Directional accuracy (unweighted): {dir_acc:.1%}")
print(f"Directional accuracy (production-weighted): {dir_acc_w:.1%}")

# ---------------------------------------------------------------------------
# 1. Predicted vs. actual scatter (log-change in area), all transitions
#    pooled, unweighted vs. production-weighted
# ---------------------------------------------------------------------------

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(8, 3))

lims = np.nanpercentile(np.concatenate([actual_a, predicted_a]), [1, 99])
pad = 0.1 * (lims[1] - lims[0])
lims = (lims[0] - pad, lims[1] + pad)

for ax, sizes, title in (
    (ax1a, 6, "Unweighted"),
    (ax1b, 4 + 200 * (weight / weight.max()), "Production-weighted (marker size)"),
):
    ax.plot(lims, lims, color="0.4", linestyle="--", linewidth=1, zorder=1)
    ax.scatter(actual_a[correct_direction], predicted_a[correct_direction],
               s=sizes if np.isscalar(sizes) else sizes[correct_direction],
               color=COLOR_CORRECT, alpha=0.35, edgecolors="none", label="correct direction",
               zorder=2, rasterized=True)
    ax.scatter(actual_a[~correct_direction], predicted_a[~correct_direction],
               s=sizes if np.isscalar(sizes) else sizes[~correct_direction],
               color=COLOR_WRONG, alpha=0.35, edgecolors="none", label="wrong direction",
               zorder=2, rasterized=True)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.set_xlabel("Actual Delta-log(area harvested)")
    ax.set_ylabel("Beta-predicted Delta-log(area harvested)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

ax1a.text(0.97, 0.03, f"unweighted acc. {dir_acc:.0%}, R2 {r2:.2f}", transform=ax1a.transAxes,
          ha="right", va="bottom", fontsize=8, color="0.3")
ax1b.text(0.97, 0.03, f"weighted acc. {dir_acc_w:.0%}", transform=ax1b.transAxes,
          ha="right", va="bottom", fontsize=8, color="0.3")

fig1.tight_layout()

# ---------------------------------------------------------------------------
# 2. Prediction-error distribution, 3x2: one row per scenario, one column
#    per unweighted / production-weighted -- shared bin range across all 6
#    panels so the rows are visually comparable.
# ---------------------------------------------------------------------------

clip_lo, clip_hi = np.nanpercentile(trans["error_pct"], [1, 99])
hist_range = (clip_lo, clip_hi)

fig2, axes2 = plt.subplots(3, 2, figsize=(8, 8.6))

for row, key in enumerate(SCENARIO_KEYS):
    sub = trans[trans["scenario"] == key]
    err = sub["error_pct"].to_numpy(float)
    w = sub["Production"].to_numpy(float)
    ax_u, ax_w = axes2[row]

    ax_u.hist(err, bins=60, range=hist_range, color=BASE_PALETTE[0], alpha=0.8)
    ax_u.axvline(0, color="0.3", linewidth=1)
    ax_u.axvline(np.mean(err), color=BASE_PALETTE[7], linewidth=1.5, label=f"mean {np.mean(err):.1f}%")
    ax_u.axvline(np.median(err), color=BASE_PALETTE[5], linewidth=1.5, label=f"median {np.median(err):.1f}%")
    ax_u.set_ylabel("Number of transitions")
    ax_u.set_title(f"{SCENARIO_LABELS[key]} \nunweighted ({len(sub)} transitions)", fontsize=9)
    ax_u.legend(fontsize=8)

    w_mean = np.average(err, weights=w)
    ax_w.hist(err, bins=60, range=hist_range, weights=w, color=BASE_PALETTE[2], alpha=0.8)
    ax_w.axvline(0, color="0.3", linewidth=1)
    ax_w.axvline(w_mean, color=BASE_PALETTE[7], linewidth=1.5, label=f"weighted mean {w_mean:.1f}%")
    ax_w.set_ylabel("Production (summed)")
    ax_w.set_title(f"{SCENARIO_LABELS[key]} \nproduction-weighted", fontsize=9)
    ax_w.legend(fontsize=8)

    if row == 2:
        ax_u.set_xlabel("Prediction error \n(actual - predicted, % of area)")
        ax_w.set_xlabel("Prediction error \n(actual - predicted, % of area)")

fig2.tight_layout()

# ---------------------------------------------------------------------------
# 3. Top items BY CURRENT PRODUCTION (not by error), grouped bars of
#    production-weighted mean |% error| per scenario.
# ---------------------------------------------------------------------------

pct_item_df = trans[["Item", "scenario", "error_pct", "Production"]].copy()
pct_item_df["abs_error_pct"] = pct_item_df["error_pct"].abs()

wm = {}
for key in SCENARIO_KEYS:
    sub = pct_item_df[pct_item_df["scenario"] == key]
    wm[key] = weighted_group_mean(sub, "Item", "abs_error_pct", "Production", min_count=MIN_SERIES_PER_ITEM)

qualifying_items = set(wm["a"].index) & set(wm["b"].index) & set(wm["c"].index)
production_by_item = (
    series[series["Item"].isin(qualifying_items)]
    .groupby("Item")["current_production"].sum().sort_values(ascending=False)
)
top_items = production_by_item.head(TOP_N_ITEMS).index.tolist()
plot_items = top_items[::-1]  # most-produced at the top of the barh

y = np.arange(len(plot_items))
bar_h = 0.25
fig3, ax3 = plt.subplots(figsize=(6, 0.6 * len(plot_items) + 1))
for i, key in enumerate(SCENARIO_KEYS):
    offset = (i - 1) * bar_h
    values = wm[key].reindex(plot_items).values
    ax3.barh(y + offset, values, height=bar_h, color=SCENARIO_COLORS[i], label=SCENARIO_LABELS[key])
ax3.set_yticks(y)
ax3.set_yticklabels(plot_items)
ax3.set_xlabel("Production-weighted mean |prediction error| (% of actual area)")
ax3.legend(fontsize=8)
fig3.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "land_use_pred_vs_actual_scatter.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "land_use_pred_vs_actual_error_distribution.png", dpi=300, bbox_inches="tight")
    fig3.savefig(figs_path / "land_use_pred_vs_actual_by_item.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved figures to {figs_path}")
