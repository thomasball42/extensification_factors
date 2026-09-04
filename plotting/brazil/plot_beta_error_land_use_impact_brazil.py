"""
Land-use impact of the model's out-of-sample estimation error for Brazil's
local (immediate-region) pasture panel -- the Brazil analog of
../plot_beta_error_land_use_impact_crops.py. Uses the same extensification
projection formula the pipeline exists to support:

    A_n = A_t * ((P_t + X) / P_t) ** beta = A_t * (1 + f) ** beta

where A_t/P_t are current pasture area/production
(../../brazil/outputs/beta_animals_local_Brazil.csv) and f is an assumed
fractional increase in production (X = f * P_t). The "error" applied to A_n
is the model's own validated out-of-sample one-step prediction error,
mae_tvp, from validation/brazil/holdout_comparison_animals_brazil.py's held-
out cross-validation (in Delta-log(area) "log points") -- NOT the in-sample
smoother SE, for the same reason as the crops version: the holdout MAE is
the empirically demonstrated error rather than a model-internal uncertainty
estimate.

Only regions with BOTH a current beta and a validated holdout MAE are
included -- holdout validation requires longer series (MIN_TRAIN_OBS=6 +
HOLDOUT_YEARS=2) than the main pipeline (MIN_OBS=8), so coverage is a subset
of beta_animals_local_Brazil.csv.

Brazil has no crop dimension (a single pooled "All pasture-based animal
products" series per region), so the crops script's per-item "top items"
panel becomes a per-state (uf) panel instead -- states aggregate many
regions the same way crop items aggregate many countries. All ~22 states fit
on one bar chart with no top-N truncation needed, and the sensitivity-sweep
panel has a single "All regions" line (no per-item overlay, since there's
only one series type).

Produces (saved to ../../figs/brazil/):
  land_use_error_distribution_brazil.png  -- headline distribution of the
                                              relative land-use error (%) at
                                              one benchmark production-
                                              increase scenario, unweighted
                                              vs. production-weighted
  land_use_error_sweep_brazil.png         -- total hectares of uncertainty
                                              vs. the size of the assumed
                                              production increase
  land_use_error_by_uf_brazil.png         -- states (uf) by production-
                                              weighted mean % error at the
                                              benchmark scenario

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
HOLDOUT_PATH = Path("outputs") / "validation" / "brazil" / "holdout_comparison_animals_brazil.csv"
figs_path = Path("..") / "figs" / "brazil"
SAVE = True

BENCHMARK_F = 0.20   # headline production-increase scenario (+20%), matches the crops script
SWEEP_F_MAX = 1.0    # sensitivity sweep goes from 0% to +100% production
SWEEP_N = 50
MIN_REGIONS_PER_UF = 3  # drop states backed by too few regions to be a meaningful ranking

AREA_COL = "current_area_pasture_ha"
PROD_COL = "current_production_kg"

BASE_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

os.makedirs(figs_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Load + prep
# ---------------------------------------------------------------------------

beta_df = pd.read_csv(BETA_PATH)
beta_df = add_beta_current(beta_df)
beta_df = beta_df.dropna(subset=["beta_current", AREA_COL, PROD_COL])
beta_df = beta_df[(beta_df[AREA_COL] > 0) & (beta_df[PROD_COL] > 0)]

holdout_df = pd.read_csv(HOLDOUT_PATH)[["cod_rgi", "mae_tvp", "test_avg_production"]]

df = beta_df.merge(holdout_df, on="cod_rgi", how="inner")
print(f"{len(df)} regions with both a current beta and a validated holdout MAE "
      f"(of {len(beta_df)} regions in {BETA_PATH.name}).")

A_t = df[AREA_COL].to_numpy(float)
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
print(f"  Total hectares of uncertainty summed across all {len(df)} regions: "
      f"{error_ha_bench.sum() / 1e6:.2f} million ha")

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
ax1a.set_ylabel("Number of regions")
ax1a.set_title("Unweighted (every region counts equally)")
ax1a.legend()

w_mean = np.average(error_pct_clipped, weights=weight)
ax1b.hist(error_pct_clipped, bins=60, weights=weight, color=BASE_PALETTE[2], alpha=0.8)
ax1b.axvline(w_mean, color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
             label=f"weighted mean {w_mean:.1f}%")
ax1b.set_xlabel(f"Land-use projection error (%), at a +{BENCHMARK_F:.0%} production scenario")
ax1b.set_ylabel("Production (current, summed)")
ax1b.set_title("Production-weighted")
ax1b.legend()

fig1.suptitle(f"Brazil: what does the model's validated out-of-sample error mean for a\n"
              f"pasture land-use projection? ({len(df)} regions, +{BENCHMARK_F:.0%} production scenario)")
fig1.tight_layout()

# ---------------------------------------------------------------------------
# 2. Sensitivity sweep -- single "All regions" line (no item dimension)
# ---------------------------------------------------------------------------

f_values = np.linspace(0, SWEEP_F_MAX, SWEEP_N)
total_error_ha = np.empty(SWEEP_N)
for i, f in enumerate(f_values):
    _, error_ha, _ = land_use_projection(f)
    total_error_ha[i] = error_ha.sum()

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(f_values * 100, total_error_ha / 1e6, color=BASE_PALETTE[0], linewidth=2.5, label="All regions")
ax2.axvline(BENCHMARK_F * 100, color="0.5", linestyle="--", linewidth=1)
ax2.set_xlabel("Assumed production increase (%)")
ax2.set_ylabel("Total land-use projection uncertainty (million ha)")
ax2.set_title(f"Brazil: how the land-use stakes of the model's error grow with the\n"
              f"size of the assumed production-increase scenario ({len(df)} regions)")
ax2.legend(loc="upper left", fontsize=8)
fig2.tight_layout()

# ---------------------------------------------------------------------------
# 3. States (uf) by production-weighted mean % error
#
# Ranking by total hectares of uncertainty (as in panel 2) is dominated by
# whichever states simply have the largest pasture footprint. This panel
# instead ranks by production-weighted mean % error, mirroring the crops
# script's by-item fix -- each region weighted by its held-out-test-year
# average production.
# ---------------------------------------------------------------------------

uf_df = pd.DataFrame({"uf": df["uf"].values, "error_pct_bench": error_pct_bench, "weight": weight})
by_uf_pct = weighted_group_mean(uf_df, "uf", "error_pct_bench", "weight", min_count=MIN_REGIONS_PER_UF).iloc[::-1]

fig3, ax3 = plt.subplots(figsize=(8, 0.4 * len(by_uf_pct) + 2))
ax3.barh(np.arange(len(by_uf_pct)), by_uf_pct.values, color=BASE_PALETTE[0])
ax3.set_yticks(np.arange(len(by_uf_pct)))
ax3.set_yticklabels(by_uf_pct.index)
ax3.set_xlabel("Production-weighted mean land-use projection error (%)")
ax3.set_title(f"Brazil: states (uf) by production-weighted % land-use error, "
              f"+{BENCHMARK_F:.0%} production scenario\n(states with < {MIN_REGIONS_PER_UF} regions excluded)")
fig3.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "land_use_error_distribution_brazil.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "land_use_error_sweep_brazil.png", dpi=300, bbox_inches="tight")
    fig3.savefig(figs_path / "land_use_error_by_uf_brazil.png", dpi=300, bbox_inches="tight")

print(f"\nSaved figures to {figs_path}")
