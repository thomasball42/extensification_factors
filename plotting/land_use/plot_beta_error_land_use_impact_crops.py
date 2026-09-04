"""
Land-use impact of the model's out-of-sample estimation error: how far off
could a beta-based land-use projection actually be, in hectares?

Uses the extensification projection formula the pipeline exists to support:

    A_n = A_t * ((P_t + X) / P_t) ** beta = A_t * (1 + f) ** beta

where A_t/P_t are current area/production (outputs/beta_crops.csv) and f is
an assumed fractional increase in production (X = f * P_t). The "error"
applied to A_n is the model's own validated out-of-sample one-step
prediction error, mae_tvp, from validation/holdout_comparison.py's held-out
cross-validation (in Delta-log(Area) "log points") -- NOT the in-sample
smoother SE, since the holdout MAE is the empirically demonstrated error
rather than a model-internal uncertainty estimate. Applying it
multiplicatively to A_n (A_n * exp(+-mae_tvp)) turns that abstract
log-point error into a concrete hectare/percent uncertainty band on the
land-use projection -- the "MAE, but for land-use values" this quantifies.

Only series with BOTH a current beta (beta_crops.csv) and a validated
holdout MAE (holdout_static_comparison_pweighted.csv) are included --
holdout validation requires longer series (MIN_TRAIN_OBS=30 +
HOLDOUT_YEARS=5 years, see validation/holdout_comparison.py) than the main
pipeline (MIN_OBS=15), so coverage is a strict subset of beta_crops.csv.

Produces:
  land_use_error_distribution.png  -- headline distribution of the relative
                                       land-use error (%) at one benchmark
                                       production-increase scenario, unweighted
                                       vs. production-weighted
  land_use_error_sweep.png         -- total hectares of uncertainty vs. the
                                       size of the assumed production increase
  land_use_error_by_item.png       -- top items by production-weighted mean
                                       % error at the benchmark scenario (not
                                       total hectares -- that ranking is
                                       dominated by whichever crops simply
                                       have the largest land footprint)
"""

import colorsys
import os
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _utils import (
    add_beta_current, filter_list as _filter_list, load_aggregate_matcher,
    load_plot_config, weighted_group_mean,
)

SCRIPT_NAME = "plot_beta_error_land_use_impact_crops"

BETA_PATH = Path("outputs") / "beta_crops.csv"
HOLDOUT_PATH = Path("outputs") / "validation" / "holdout_static_comparison_pweighted.csv"
figs_path = Path("..") / "figs" / "land_use" / "beta_error_land_use_impact_crops"
SAVE = True

_lui_cfg = load_plot_config()["land_use_impact"][SCRIPT_NAME]
BENCHMARK_F = _lui_cfg["benchmark_f"]   # headline production-increase scenario (+20%)
SWEEP_F_MAX = _lui_cfg["sweep_f_max"]   # sensitivity sweep goes from 0% to +100% production
SWEEP_N = _lui_cfg["sweep_n"]
TOP_N_ITEMS = _lui_cfg["top_n_items"]
MIN_SERIES_PER_ITEM = _lui_cfg.get("min_series_per_item", 3)

ifilt = _lui_cfg.get("ifilt", [])  # empty = no item filtering (every crop with a validated holdout MAE)
iexcl = _lui_cfg.get("iexcl", [])

BASE_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]


def distinct_colors(n):
    """n visually distinct colors: the hand-picked base palette for small n,
    golden-angle-spaced HSV hues (with alternating sat/val) beyond that so
    consecutive categories never land on near-identical hues."""
    if n <= len(BASE_PALETTE):
        return BASE_PALETTE[:n]
    colors = list(BASE_PALETTE)
    for i in range(len(BASE_PALETTE), n):
        hue = (i * 0.6180339887) % 1.0
        sat = 0.55 if (i // len(BASE_PALETTE)) % 2 else 0.85
        val = 0.65 if (i // len(BASE_PALETTE)) % 2 else 0.9
        colors.append(mcolors.to_hex(colorsys.hsv_to_rgb(hue, sat, val)))
    return colors


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

holdout_df = pd.read_csv(HOLDOUT_PATH)[["Area", "Item", "Item Code", "mae_tvp", "test_avg_production"]]

df = beta_df.merge(holdout_df, on=["Area", "Item", "Item Code"], how="inner")
print(f"{len(df)} series with both a current beta and a validated holdout MAE "
      f"(of {len(beta_df)} non-aggregate series in {BETA_PATH.name}).")

A_t = df[area_col].to_numpy(float) * ha_multiplier
beta = df["beta_current"].to_numpy(float)
mae = df["mae_tvp"].to_numpy(float)
weight = df["test_avg_production"].to_numpy(float)


def land_use_projection(f):
    """A_n, error_ha, error_pct for production-increase scenario f (fraction
    of current production, X = f * P_t), applying the model's out-of-sample
    MAE multiplicatively around the point-estimate projection."""
    ratio = 1.0 + f
    A_n = A_t * ratio ** beta
    A_n_hi = A_n * np.exp(mae)
    A_n_lo = A_n * np.exp(-mae)
    error_ha = (A_n_hi - A_n_lo) / 2
    error_pct = 100 * (np.exp(mae) - 1)
    return A_n, error_ha, error_pct


_, error_ha_bench, error_pct_bench = land_use_projection(BENCHMARK_F)

print(f"\nAt a +{BENCHMARK_F:.0%} production-increase scenario:")
print(f"  Relative land-use error (unweighted mean/median): "
      f"{error_pct_bench.mean():.1f}% / {np.median(error_pct_bench):.1f}%")
print(f"  Relative land-use error (production-weighted mean): "
      f"{np.average(error_pct_bench, weights=weight):.1f}%")
print(f"  Total hectares of uncertainty summed across all {len(df)} series: "
      f"{error_ha_bench.sum() / 1e6:.2f} million {ha_unit}")

# ---------------------------------------------------------------------------
# 1. Headline distribution: relative land-use error at the benchmark scenario
# ---------------------------------------------------------------------------

clip_hi = np.nanpercentile(error_pct_bench, 99)
error_pct_clipped = np.clip(error_pct_bench, 0, clip_hi)

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))

ax1a.hist(error_pct_clipped, bins=60, color=BASE_PALETTE[0], alpha=0.8)
ax1a.axvline(error_pct_bench.mean(), color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
             label=f"mean {error_pct_bench.mean():.1f}%")
ax1a.axvline(np.median(error_pct_bench), color=BASE_PALETTE[5], linestyle="-", linewidth=1.5,
             label=f"median {np.median(error_pct_bench):.1f}%")
ax1a.set_xlabel(f"Land-use projection error (%), at a +{BENCHMARK_F:.0%} production scenario")
ax1a.set_ylabel("Number of series")
ax1a.set_title("Unweighted (every series counts equally)")
ax1a.legend()

w_mean = np.average(error_pct_clipped, weights=weight)
ax1b.hist(error_pct_clipped, bins=60, weights=weight, color=BASE_PALETTE[2], alpha=0.8)
ax1b.axvline(w_mean, color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
             label=f"weighted mean {w_mean:.1f}%")
ax1b.set_xlabel(f"Land-use projection error (%), at a +{BENCHMARK_F:.0%} production scenario")
ax1b.set_ylabel("Production (current, summed)")
ax1b.set_title("Production-weighted")
ax1b.legend()

fig1.suptitle(f"What does the model's validated out-of-sample error mean for a land-use "
              f"projection?\n({len(df)} crop series, +{BENCHMARK_F:.0%} production scenario)")
fig1.tight_layout()

# ---------------------------------------------------------------------------
# 2 & 3. Sensitivity sweep + top items, both ranked by benchmark-scenario error_ha
# ---------------------------------------------------------------------------

by_item_bench = (
    df.assign(error_ha_bench=error_ha_bench)
    .groupby("Item")["error_ha_bench"].sum()
    .sort_values(ascending=False)
)
top_items = by_item_bench.head(TOP_N_ITEMS).index.tolist()
item_masks = {item: (df["Item"] == item).to_numpy() for item in top_items}
item_colors = dict(zip(top_items, distinct_colors(len(top_items))))

f_values = np.linspace(0, SWEEP_F_MAX, SWEEP_N)
total_error_ha = np.empty(SWEEP_N)
item_error_ha = {item: np.empty(SWEEP_N) for item in top_items}
for i, f in enumerate(f_values):
    _, error_ha, _ = land_use_projection(f)
    total_error_ha[i] = error_ha.sum()
    for item in top_items:
        item_error_ha[item][i] = error_ha[item_masks[item]].sum()

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(f_values * 100, total_error_ha / 1e6, color="0.2", linewidth=2.5, label="All series")
for item in top_items:
    ax2.plot(f_values * 100, item_error_ha[item] / 1e6, color=item_colors[item],
              linewidth=1.2, alpha=0.85, label=item)

ax2.axvline(BENCHMARK_F * 100, color="0.5", linestyle="--", linewidth=1)
ax2.set_xlabel("Assumed production increase (%)")
ax2.set_ylabel(f"Total land-use projection uncertainty (million {ha_unit})")
ax2.set_title(f"How the land-use stakes of the model's error grow with the\n"
              f"size of the assumed production-increase scenario ({len(df)} series)")
ax2.legend(loc="upper left", fontsize=8, ncol=2)
fig2.tight_layout()

# Ranking by total hectares of uncertainty (as above, used for the sweep
# overlay) is dominated by whichever crops simply have the largest land
# footprint. This panel instead ranks by production-weighted mean % error --
# each series weighted by its held-out-test-year average production, so
# accuracy is compared in proportional terms and a handful of small, noisy
# series can't dominate an item's score.
pct_item_df = pd.DataFrame({"Item": df["Item"].values, "error_pct_bench": error_pct_bench, "weight": weight})
by_item_pct = weighted_group_mean(pct_item_df, "Item", "error_pct_bench", "weight", min_count=MIN_SERIES_PER_ITEM)

by_item_plot = by_item_pct.head(TOP_N_ITEMS).iloc[::-1]

fig3, ax3 = plt.subplots(figsize=(8, 0.4 * len(by_item_plot) + 2))
ax3.barh(np.arange(len(by_item_plot)), by_item_plot.values, color=BASE_PALETTE[0])
ax3.set_yticks(np.arange(len(by_item_plot)))
ax3.set_yticklabels(by_item_plot.index)
ax3.set_xlabel("Production-weighted mean land-use projection error (%)")
ax3.set_title(f"Top {len(by_item_plot)} items by production-weighted % land-use error, "
              f"+{BENCHMARK_F:.0%} production scenario\n(items with < {MIN_SERIES_PER_ITEM} series excluded)")
fig3.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "land_use_error_distribution.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "land_use_error_sweep.png", dpi=300, bbox_inches="tight")
    fig3.savefig(figs_path / "land_use_error_by_item.png", dpi=300, bbox_inches="tight")