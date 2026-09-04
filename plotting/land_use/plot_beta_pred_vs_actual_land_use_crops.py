"""
Validates beta against real history: for each series' most recent year (Y),
does the beta-implied land-use change from the ACTUALLY observed production
change match the area change that ACTUALLY happened?

    A_Y_predicted = A_{Y-1} * (P_Y / P_{Y-1}) ** beta_Y

compared against the actually observed A_Y. This mirrors the simplifying
assumption of plot_beta_error_land_use_impact_crops.py's own hypothetical-
scenario projection (no correction for the small series-mean demeaning
offset that beta was actually fit net of) -- so the two scripts stay
conceptually consistent, at the cost of ignoring each series' own secular
area/production trend not explained by beta.

Unlike plot_beta_error_land_use_impact_crops.py (which projects a
hypothetical production-increase scenario and brackets it with the model's
validated holdout MAE), this script uses no hypothetical scenario and no
holdout-validation data at all -- just beta_crops.csv's own current_year
snapshot plus the ONE prior year of raw Area harvested / Production pulled
from the FAOSTAT source, so it can compare a real prediction to a real
outcome.

Produces:
  land_use_pred_vs_actual_scatter.png     -- predicted vs. actual log-change
                                              in area, y=x line, unweighted
                                              vs. production-weighted
  land_use_pred_vs_actual_error_hist.png  -- distribution of the prediction
                                              error (actual - predicted),
                                              unweighted vs. production-weighted
  land_use_pred_vs_actual_by_item.png     -- top items by production-weighted
                                              mean absolute % error (not
                                              total hectares -- that ranking
                                              is dominated by whichever crops
                                              simply have the largest land
                                              footprint, not the worst
                                              predictions)
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _utils import (
    add_beta_current, filter_list as _filter_list, load_aggregate_matcher,
    load_plot_config, weighted_group_mean,
)

SCRIPT_NAME = "plot_beta_pred_vs_actual_land_use_crops"

BETA_PATH = Path("outputs") / "beta_crops.csv"
RAW_DATA_PATH = Path("data") / "inputs" / "Production_Crops_Livestock_E_All_Data_(Normalized).csv"
figs_path = Path("..") / "figs" / "land_use" / "beta_pred_vs_actual_land_use_crops"
SAVE = True

_cfg = load_plot_config()["land_use_impact"][SCRIPT_NAME]
TOP_N_ITEMS = _cfg["top_n_items"]
MIN_SERIES_PER_ITEM = _cfg.get("min_series_per_item", 3)

ifilt = _cfg.get("ifilt", [])
iexcl = _cfg.get("iexcl", [])

BASE_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
COLOR_CORRECT = "#1baf7a"
COLOR_WRONG = "#e34948"

os.makedirs(figs_path, exist_ok=True)
is_aggregate = load_aggregate_matcher()

# ---------------------------------------------------------------------------
# Load + prep
# ---------------------------------------------------------------------------

beta_df = pd.read_csv(BETA_PATH)
beta_df = beta_df[~beta_df["is_aggregate"]].copy()

if ifilt:
    items = _filter_list(beta_df["Item"].unique(), ifilt, iexcl)
    beta_df = beta_df[beta_df["Item"].isin(items)].copy()

area_col = next(c for c in beta_df.columns if c.startswith("current_area_harvested_"))
prod_col = next(c for c in beta_df.columns if c.startswith("current_production_"))
ha_unit_raw = area_col.split("current_area_harvested_", 1)[1]
ha_multiplier = 1000.0 if ha_unit_raw.strip().startswith("1000") else 1.0
ha_unit = ha_unit_raw.split()[-1]

beta_df = add_beta_current(beta_df)
beta_df = beta_df.dropna(subset=["beta_current", area_col, prod_col])
beta_df = beta_df[(beta_df[area_col] > 0) & (beta_df[prod_col] > 0)]
beta_df["prev_year"] = beta_df["current_year"].astype(int) - 1

# raw prior-year Area harvested / Production, to compare beta's prediction
# against what actually happened
raw = pd.read_csv(RAW_DATA_PATH, encoding="latin-1", low_memory=False,
                   usecols=["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value"])
raw = raw[raw["Element"].isin(["Area harvested", "Production"])]
raw_wide = raw.pivot_table(
    index=["Area", "Area Code", "Item", "Item Code", "Year"], columns="Element", values="Value"
).reset_index()
raw_wide = raw_wide.rename(columns={
    "Area harvested": "area_prev_raw", "Production": "prod_prev", "Year": "prev_year",
})

df = beta_df.merge(
    raw_wide[["Area", "Area Code", "Item", "Item Code", "prev_year", "area_prev_raw", "prod_prev"]],
    on=["Area", "Area Code", "Item", "Item Code", "prev_year"], how="inner",
)
df["area_prev"] = df["area_prev_raw"] * ha_multiplier
df = df[(df["area_prev"] > 0) & (df["prod_prev"] > 0)]
print(f"{len(df)} series with a current beta and real prior-year (Y-1) Area harvested/Production "
      f"(of {len(beta_df)} non-aggregate series with a current beta in {BETA_PATH.name}).")

# ---------------------------------------------------------------------------
# Predicted vs. actual
# ---------------------------------------------------------------------------

A_prev = df["area_prev"].to_numpy(float)
A_Y = df[area_col].to_numpy(float) * ha_multiplier
P_prev = df["prod_prev"].to_numpy(float)
P_Y = df[prod_col].to_numpy(float)
beta = df["beta_current"].to_numpy(float)
weight = df[prod_col].to_numpy(float)

actual_a = np.log(A_Y / A_prev)
predicted_a = beta * np.log(P_Y / P_prev)
prod_ratio = P_Y / P_prev

predicted_A_Y = A_prev * prod_ratio ** beta
actual_delta_ha = A_Y - A_prev
predicted_delta_ha = predicted_A_Y - A_prev
error_ha = actual_delta_ha - predicted_delta_ha

correct_direction = np.sign(actual_delta_ha) == np.sign(predicted_delta_ha)

r = np.corrcoef(actual_a, predicted_a)[0, 1]
r2 = r ** 2
dir_acc = correct_direction.mean()
dir_acc_w = np.average(correct_direction, weights=weight)

print(f"\nR-squared (predicted vs. actual log-change in area): {r2:.3f}")
print(f"Directional accuracy (unweighted): {dir_acc:.1%}")
print(f"Directional accuracy (production-weighted): {dir_acc_w:.1%}")
print(f"Mean/median absolute log-diff error: {np.mean(np.abs(actual_a - predicted_a)):.3f} / "
      f"{np.median(np.abs(actual_a - predicted_a)):.3f}")
print(f"Mean/median absolute hectare error: {np.mean(np.abs(error_ha)):,.0f} / "
      f"{np.median(np.abs(error_ha)):,.0f} {ha_unit}")

# ---------------------------------------------------------------------------
# 1. Predicted vs. actual scatter (log-change in area), unweighted vs. weighted
# ---------------------------------------------------------------------------

# fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5.5))

# lims = np.nanpercentile(np.concatenate([actual_a, predicted_a]), [1, 99])
# pad = 0.1 * (lims[1] - lims[0])
# lims = (lims[0] - pad, lims[1] + pad)

# for ax, sizes, title in (
#     (ax1a, 14, "Unweighted"),
#     (ax1b, 8 + 400 * (weight / weight.max()), "Production-weighted (marker size)"),
# ):
#     ax.plot(lims, lims, color="0.4", linestyle="--", linewidth=1, zorder=1)
#     ax.scatter(actual_a[correct_direction], predicted_a[correct_direction], s=sizes if np.isscalar(sizes) else sizes[correct_direction],
#                color=COLOR_CORRECT, alpha=0.55, edgecolors="none", label="correct direction", zorder=2)
#     ax.scatter(actual_a[~correct_direction], predicted_a[~correct_direction], s=sizes if np.isscalar(sizes) else sizes[~correct_direction],
#                color=COLOR_WRONG, alpha=0.55, edgecolors="none", label="wrong direction", zorder=2)
#     ax.set_xlim(lims)
#     ax.set_ylim(lims)
#     ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
#     ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
#     ax.set_xlabel("Actual Delta-log(Area)")
#     ax.set_ylabel("Beta-predicted Delta-log(Area)")
#     ax.set_title(title)
#     ax.legend(loc="upper left", fontsize=8)

# acc_label = f"unweighted acc. {dir_acc:.0%}, R2 {r2:.2f}"
# acc_label_w = f"weighted acc. {dir_acc_w:.0%}"
# ax1a.text(0.97, 0.03, acc_label, transform=ax1a.transAxes, ha="right", va="bottom", fontsize=8, color="0.3")
# ax1b.text(0.97, 0.03, acc_label_w, transform=ax1b.transAxes, ha="right", va="bottom", fontsize=8, color="0.3")

# fig1.suptitle(f"Does beta predict the area change that actually happened?\n"
#               f"({len(df)} crop series, each at its own most recent year)")
# fig1.tight_layout()

# # ---------------------------------------------------------------------------
# # 2. Prediction-error distribution (actual - predicted), unweighted vs. weighted
# # ---------------------------------------------------------------------------

# error_k = error_ha / 1e3
# clip_lo, clip_hi = np.nanpercentile(error_k, [1, 99])
# hist_range = (clip_lo, clip_hi)
# n_clipped = int(((error_k < clip_lo) | (error_k > clip_hi)).sum())

# fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

# ax2a.hist(error_k, bins=60, range=hist_range, color=BASE_PALETTE[0], alpha=0.8)
# ax2a.axvline(0, color="0.3", linestyle="-", linewidth=1)
# ax2a.axvline(np.mean(error_k), color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
#              label=f"mean {np.mean(error_k):,.0f}k {ha_unit}")
# ax2a.axvline(np.median(error_k), color=BASE_PALETTE[5], linestyle="-", linewidth=1.5,
#              label=f"median {np.median(error_k):,.0f}k {ha_unit}")
# ax2a.set_xlabel(f"Prediction error, actual - predicted (1000 {ha_unit})\n"
#                  f"[{n_clipped} series outside the 1st-99th percentile range not shown]")
# ax2a.set_ylabel("Number of series")
# ax2a.set_title("Unweighted")
# ax2a.legend(fontsize=8)

# w_mean = np.average(error_k, weights=weight)
# ax2b.hist(error_k, bins=60, range=hist_range, weights=weight, color=BASE_PALETTE[2], alpha=0.8)
# ax2b.axvline(0, color="0.3", linestyle="-", linewidth=1)
# ax2b.axvline(w_mean, color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
#              label=f"weighted mean {w_mean:,.0f}k {ha_unit}")
# ax2b.set_xlabel(f"Prediction error, actual - predicted (1000 {ha_unit})\n"
#                  f"[{n_clipped} series outside the 1st-99th percentile range not shown]")
# ax2b.set_ylabel("Production (current, summed)")
# ax2b.set_title("Production-weighted")
# ax2b.legend(fontsize=8)

# fig2.suptitle(f"How far off is beta's predicted land-use change from what actually happened?\n"
#               f"({len(df)} crop series)")
# fig2.tight_layout()

# ---------------------------------------------------------------------------
# Top items by production-weighted mean absolute % error
#
# Ranking by total hectares of miss (as before) is dominated by whichever
# crops simply have the largest land footprint, regardless of how accurate
# the prediction was relative to that crop's own size. Ranking by
# production-weighted mean |error| / actual area instead measures accuracy
# in proportional terms, with each series weighted by its current
# production so a handful of small, noisy series can't dominate an item's
# score.
# ---------------------------------------------------------------------------

pct_error = 100 * np.abs(error_ha) / A_Y
item_df = pd.DataFrame({"Item": df["Item"].values, "pct_error": pct_error, "weight": weight})
by_item_pct = weighted_group_mean(item_df, "Item", "pct_error", "weight", min_count=MIN_SERIES_PER_ITEM)

by_item_plot = by_item_pct.head(TOP_N_ITEMS).iloc[::-1]

fig3, ax3 = plt.subplots(figsize=(8, 0.4 * len(by_item_plot) + 2))
ax3.barh(np.arange(len(by_item_plot)), by_item_plot.values, color=BASE_PALETTE[0])
ax3.set_yticks(np.arange(len(by_item_plot)))
ax3.set_yticklabels(by_item_plot.index)
ax3.set_xlabel("Production-weighted mean absolute % error in predicted land area")
ax3.set_title(f"Top {len(by_item_plot)} items by production-weighted % "
              f"land-use prediction miss\n(items with < {MIN_SERIES_PER_ITEM} series excluded)")
fig3.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "land_use_pred_vs_actual_scatter.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "land_use_pred_vs_actual_error_hist.png", dpi=300, bbox_inches="tight")
    fig3.savefig(figs_path / "land_use_pred_vs_actual_by_item.png", dpi=300, bbox_inches="tight")
