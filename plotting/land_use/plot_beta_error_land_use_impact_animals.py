"""
Animals counterpart to plot_beta_error_land_use_impact_crops.py -- see that
module's docstring for the full methodology (same formula, same use of the
out-of-sample holdout MAE as the error source). Reads outputs/beta_animals.csv
and outputs/validation/holdout_static_comparison_animals_pweighted.csv
(validation/holdout_comparison_animals.py) instead of the crops files.

Unlike crops, there is only one pooled "item" here (all pasture-based
animal products per Area -- see _compute_beta_animals.py), so there's no
per-item breakdown panel and no is_aggregate filter to apply.

Produces:
  land_use_error_distribution.png  -- headline distribution of the relative
                                       land-use error (%) at one benchmark
                                       production-increase scenario, unweighted
                                       vs. production-weighted
  land_use_error_sweep.png         -- total hectares of uncertainty vs. the
                                       size of the assumed production increase
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from _utils import load_plot_config

SCRIPT_NAME = "plot_beta_error_land_use_impact_animals"

BETA_PATH = Path("outputs") / "beta_animals.csv"
HOLDOUT_PATH = Path("outputs") / "validation" / "holdout_static_comparison_animals_pweighted.csv"
figs_path = Path("..") / "figs" / "land_use" / "beta_error_land_use_impact_animals"
SAVE = True

_lui_cfg = load_plot_config()["land_use_impact"][SCRIPT_NAME]
BENCHMARK_F = _lui_cfg["benchmark_f"]   # headline production-increase scenario (+20%)
SWEEP_F_MAX = _lui_cfg["sweep_f_max"]   # sensitivity sweep goes from 0% to +100% production
SWEEP_N = _lui_cfg["sweep_n"]

BASE_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

os.makedirs(figs_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Load + prep
# ---------------------------------------------------------------------------

beta_df = pd.read_csv(BETA_PATH)

area_col = next(c for c in beta_df.columns if c.startswith("current_area_harvested_"))
prod_col = next(c for c in beta_df.columns if c.startswith("current_production_"))
ha_unit_raw = area_col.split("current_area_harvested_", 1)[1]
ha_multiplier = 1000.0 if ha_unit_raw.strip().startswith("1000") else 1.0
ha_unit = ha_unit_raw.split()[-1]

# each series's own most recent (current_year) beta -- not a fixed calendar year
beta_year_cols = sorted((c for c in beta_df.columns if c.startswith("beta_")), key=lambda c: int(c.split("_")[1]))
years_avail = [int(c.split("_")[1]) for c in beta_year_cols]
col_of_year = {y: i for i, y in enumerate(years_avail)}
current_year = beta_df["current_year"].astype(int)
col_pos = current_year.map(col_of_year)
has_col = col_pos.notna()
beta_df["beta_current"] = np.nan
row_pos = np.flatnonzero(has_col.values)
col_pos_valid = col_pos[has_col].astype(int).values
beta_df.loc[has_col, "beta_current"] = beta_df[beta_year_cols].values[row_pos, col_pos_valid]

beta_df = beta_df.dropna(subset=["beta_current", area_col, prod_col])
beta_df = beta_df[(beta_df[area_col] > 0) & (beta_df[prod_col] > 0)]

holdout_df = pd.read_csv(HOLDOUT_PATH)[["Area", "Item", "Item Code", "mae_tvp", "test_avg_production"]]

df = beta_df.merge(holdout_df, on=["Area", "Item", "Item Code"], how="inner")
print(f"{len(df)} areas with both a current beta and a validated holdout MAE "
      f"(of {len(beta_df)} series in {BETA_PATH.name}).")

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
print(f"  Total hectares of uncertainty summed across all {len(df)} areas: "
      f"{error_ha_bench.sum() / 1e6:.2f} million {ha_unit}")

# ---------------------------------------------------------------------------
# 1. Headline distribution: relative land-use error at the benchmark scenario
# ---------------------------------------------------------------------------

clip_hi = np.nanpercentile(error_pct_bench, 99)
error_pct_clipped = np.clip(error_pct_bench, 0, clip_hi)

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))

ax1a.hist(error_pct_clipped, bins=40, color=BASE_PALETTE[0], alpha=0.8)
ax1a.axvline(error_pct_bench.mean(), color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
             label=f"mean {error_pct_bench.mean():.1f}%")
ax1a.axvline(np.median(error_pct_bench), color=BASE_PALETTE[5], linestyle="-", linewidth=1.5,
             label=f"median {np.median(error_pct_bench):.1f}%")
ax1a.set_xlabel(f"Land-use projection error (%), at a +{BENCHMARK_F:.0%} production scenario")
ax1a.set_ylabel("Number of areas")
ax1a.set_title("Unweighted (every area counts equally)")
ax1a.legend()

w_mean = np.average(error_pct_clipped, weights=weight)
ax1b.hist(error_pct_clipped, bins=40, weights=weight, color=BASE_PALETTE[2], alpha=0.8)
ax1b.axvline(w_mean, color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
             label=f"weighted mean {w_mean:.1f}%")
ax1b.set_xlabel(f"Land-use projection error (%), at a +{BENCHMARK_F:.0%} production scenario")
ax1b.set_ylabel("Production (current, summed)")
ax1b.set_title("Production-weighted")
ax1b.legend()

fig1.suptitle(f"What does the model's validated out-of-sample error mean for a land-use "
              f"projection?\n({len(df)} pasture areas, +{BENCHMARK_F:.0%} production scenario)")
fig1.tight_layout()

# ---------------------------------------------------------------------------
# 2. Sensitivity sweep: total hectares of uncertainty vs. scenario size
# ---------------------------------------------------------------------------

f_values = np.linspace(0, SWEEP_F_MAX, SWEEP_N)
total_error_ha = np.empty(SWEEP_N)
for i, f in enumerate(f_values):
    _, error_ha, _ = land_use_projection(f)
    total_error_ha[i] = error_ha.sum()

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(f_values * 100, total_error_ha / 1e6, color=BASE_PALETTE[0], linewidth=2.5, label="All pasture areas")
ax2.axvline(BENCHMARK_F * 100, color="0.5", linestyle="--", linewidth=1)
ax2.set_xlabel("Assumed production increase (%)")
ax2.set_ylabel(f"Total land-use projection uncertainty (million {ha_unit})")
ax2.set_title(f"How the land-use stakes of the model's error grow with the\n"
              f"size of the assumed production-increase scenario ({len(df)} areas)")
ax2.legend(loc="upper left")
fig2.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "land_use_error_distribution.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "land_use_error_sweep.png", dpi=300, bbox_inches="tight")