"""
Validates beta against real history for Brazil's local (immediate-region)
pasture panel -- the Brazil analog of
../plot_beta_pred_vs_actual_land_use_crops.py. For each region's most recent
year (Y), does the beta-implied pasture-area change from the ACTUALLY
observed production change match the area change that ACTUALLY happened?

    A_Y_predicted = A_{Y-1} * (P_Y / P_{Y-1}) ** beta_Y

compared against the actually observed A_Y. Mirrors the crops script's own
simplifying assumption (no correction for the small series-mean demeaning
offset beta was actually fit net of), so the two stay conceptually
consistent.

Uses ../../brazil/outputs/beta_animals_local_Brazil.csv (current beta per
region) plus ../../brazil/data/NL_local_domain.csv (the one prior year of raw
area_past_ha / P_T per region needed to compare a real prediction to a real
outcome) -- no holdout-validation data involved, exactly like the crops
version.

Brazil has no crop dimension (a single pooled "All pasture-based animal
products" series per region), so the per-crop "top items" panel becomes a
per-state (uf) panel instead -- states aggregate many regions the same way
crop items aggregate many countries. All ~22 states fit on one bar chart with
no top-N truncation needed.

Produces (saved to ../../figs/brazil/):
  land_use_pred_vs_actual_scatter_brazil.png     -- predicted vs. actual
                                                     log-change in pasture
                                                     area, y=x line,
                                                     unweighted vs.
                                                     production-weighted
  land_use_pred_vs_actual_error_hist_brazil.png  -- distribution of the
                                                     prediction error (actual
                                                     - predicted), unweighted
                                                     vs. production-weighted
  land_use_pred_vs_actual_by_uf_brazil.png       -- states (uf) by
                                                     production-weighted mean
                                                     absolute % error

Run with working directory = extensification_factors/.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _utils import add_beta_current, weighted_group_mean

BETA_PATH = Path("..") / "brazil" / "outputs" / "beta_animals_local_Brazil.csv"
RAW_DATA_PATH = Path("..") / "brazil" / "data" / "NL_local_domain.csv"
figs_path = Path("..") / "figs" / "brazil"
SAVE = True

MIN_REGIONS_PER_UF = 3  # drop states backed by too few regions to be a meaningful ranking

AREA_COL = "current_area_pasture_ha"
PROD_COL = "current_production_kg"

BASE_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
COLOR_CORRECT = "#1baf7a"
COLOR_WRONG = "#e34948"

os.makedirs(figs_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Load + prep
# ---------------------------------------------------------------------------

beta_df = pd.read_csv(BETA_PATH)
beta_df = add_beta_current(beta_df)
beta_df = beta_df.dropna(subset=["beta_current", AREA_COL, PROD_COL])
beta_df = beta_df[(beta_df[AREA_COL] > 0) & (beta_df[PROD_COL] > 0)]
beta_df["prev_year"] = beta_df["current_year"].astype(int) - 1

# raw prior-year area_past_ha / P_T, to compare beta's prediction against
# what actually happened -- already one row per cod_rgi x year, no pivot needed
raw = pd.read_csv(RAW_DATA_PATH, encoding="utf-8", usecols=["cod_rgi", "year", "area_past_ha", "P_T"])
raw = raw.rename(columns={"year": "prev_year", "area_past_ha": "area_prev", "P_T": "prod_prev"})

df = beta_df.merge(raw, on=["cod_rgi", "prev_year"], how="inner")
df = df[(df["area_prev"] > 0) & (df["prod_prev"] > 0)]
print(f"{len(df)} regions with a current beta and real prior-year (Y-1) area/production "
      f"(of {len(beta_df)} regions with a current beta in {BETA_PATH.name}).")

# ---------------------------------------------------------------------------
# Predicted vs. actual
# ---------------------------------------------------------------------------

A_prev = df["area_prev"].to_numpy(float)
A_Y = df[AREA_COL].to_numpy(float)
P_prev = df["prod_prev"].to_numpy(float)
P_Y = df[PROD_COL].to_numpy(float)
beta = df["beta_current"].to_numpy(float)
weight = df[PROD_COL].to_numpy(float)

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

print(f"\nR-squared (predicted vs. actual log-change in pasture area): {r2:.3f}")
print(f"Directional accuracy (unweighted): {dir_acc:.1%}")
print(f"Directional accuracy (production-weighted): {dir_acc_w:.1%}")
print(f"Mean/median absolute log-diff error: {np.mean(np.abs(actual_a - predicted_a)):.3f} / "
      f"{np.median(np.abs(actual_a - predicted_a)):.3f}")
print(f"Mean/median absolute hectare error: {np.mean(np.abs(error_ha)):,.0f} / "
      f"{np.median(np.abs(error_ha)):,.0f} ha")

# ---------------------------------------------------------------------------
# 1. Predicted vs. actual scatter (log-change in area), unweighted vs. weighted
# ---------------------------------------------------------------------------

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5.5))

lims = np.nanpercentile(np.concatenate([actual_a, predicted_a]), [1, 99])
pad = 0.1 * (lims[1] - lims[0])
lims = (lims[0] - pad, lims[1] + pad)

for ax, sizes, title in (
    (ax1a, 14, "Unweighted"),
    (ax1b, 8 + 400 * (weight / weight.max()), "Production-weighted (marker size)"),
):
    ax.plot(lims, lims, color="0.4", linestyle="--", linewidth=1, zorder=1)
    ax.scatter(actual_a[correct_direction], predicted_a[correct_direction], s=sizes if np.isscalar(sizes) else sizes[correct_direction],
               color=COLOR_CORRECT, alpha=0.55, edgecolors="none", label="correct direction", zorder=2)
    ax.scatter(actual_a[~correct_direction], predicted_a[~correct_direction], s=sizes if np.isscalar(sizes) else sizes[~correct_direction],
               color=COLOR_WRONG, alpha=0.55, edgecolors="none", label="wrong direction", zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    ax.set_xlabel("Actual Delta-log(pasture area)")
    ax.set_ylabel("Beta-predicted Delta-log(pasture area)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

acc_label = f"unweighted acc. {dir_acc:.0%}, R2 {r2:.2f}"
acc_label_w = f"weighted acc. {dir_acc_w:.0%}"
ax1a.text(0.97, 0.03, acc_label, transform=ax1a.transAxes, ha="right", va="bottom", fontsize=8, color="0.3")
ax1b.text(0.97, 0.03, acc_label_w, transform=ax1b.transAxes, ha="right", va="bottom", fontsize=8, color="0.3")

fig1.suptitle(f"Brazil: does beta predict the pasture-area change that actually happened?\n"
              f"({len(df)} immediate regions, each at its own most recent year)")
fig1.tight_layout()

# ---------------------------------------------------------------------------
# 2. Prediction-error distribution (actual - predicted), unweighted vs. weighted
# ---------------------------------------------------------------------------

error_k = error_ha / 1e3
clip_lo, clip_hi = np.nanpercentile(error_k, [1, 99])
hist_range = (clip_lo, clip_hi)
n_clipped = int(((error_k < clip_lo) | (error_k > clip_hi)).sum())

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

ax2a.hist(error_k, bins=60, range=hist_range, color=BASE_PALETTE[0], alpha=0.8)
ax2a.axvline(0, color="0.3", linestyle="-", linewidth=1)
ax2a.axvline(np.mean(error_k), color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
             label=f"mean {np.mean(error_k):,.0f}k ha")
ax2a.axvline(np.median(error_k), color=BASE_PALETTE[5], linestyle="-", linewidth=1.5,
             label=f"median {np.median(error_k):,.0f}k ha")
ax2a.set_xlabel(f"Prediction error, actual - predicted (1000 ha)\n"
                 f"[{n_clipped} regions outside the 1st-99th percentile range not shown]")
ax2a.set_ylabel("Number of regions")
ax2a.set_title("Unweighted")
ax2a.legend(fontsize=8)

w_mean = np.average(error_k, weights=weight)
ax2b.hist(error_k, bins=60, range=hist_range, weights=weight, color=BASE_PALETTE[2], alpha=0.8)
ax2b.axvline(0, color="0.3", linestyle="-", linewidth=1)
ax2b.axvline(w_mean, color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
             label=f"weighted mean {w_mean:,.0f}k ha")
ax2b.set_xlabel(f"Prediction error, actual - predicted (1000 ha)\n"
                 f"[{n_clipped} regions outside the 1st-99th percentile range not shown]")
ax2b.set_ylabel("Production (current, summed)")
ax2b.set_title("Production-weighted")
ax2b.legend(fontsize=8)

fig2.suptitle(f"Brazil: how far off is beta's predicted pasture-area change from what actually happened?\n"
              f"({len(df)} immediate regions)")
fig2.tight_layout()

# ---------------------------------------------------------------------------
# 3. States (uf) by production-weighted mean absolute % error
#
# Brazil has no crop dimension to break down by -- states aggregate many
# regions the same way crop items aggregate many countries, so this plays
# the same structural role as the crops script's "top items" panel. Ranking
# by total hectares of miss would be dominated by whichever states simply
# have the largest pasture footprint; ranking by production-weighted mean
# |error| / actual area instead measures accuracy in proportional terms.
# ---------------------------------------------------------------------------

pct_error = 100 * np.abs(error_ha) / A_Y
uf_df = pd.DataFrame({"uf": df["uf"].values, "pct_error": pct_error, "weight": weight})
by_uf_pct = weighted_group_mean(uf_df, "uf", "pct_error", "weight", min_count=MIN_REGIONS_PER_UF).iloc[::-1]

fig3, ax3 = plt.subplots(figsize=(8, 0.4 * len(by_uf_pct) + 2))
ax3.barh(np.arange(len(by_uf_pct)), by_uf_pct.values, color=BASE_PALETTE[0])
ax3.set_yticks(np.arange(len(by_uf_pct)))
ax3.set_yticklabels(by_uf_pct.index)
ax3.set_xlabel("Production-weighted mean absolute % error in predicted pasture area")
ax3.set_title(f"Brazil: states (uf) by production-weighted % land-use prediction miss\n"
              f"(states with < {MIN_REGIONS_PER_UF} regions excluded)")
fig3.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "land_use_pred_vs_actual_scatter_brazil.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "land_use_pred_vs_actual_error_hist_brazil.png", dpi=300, bbox_inches="tight")
    fig3.savefig(figs_path / "land_use_pred_vs_actual_by_uf_brazil.png", dpi=300, bbox_inches="tight")

print(f"\nSaved figures to {figs_path}")
