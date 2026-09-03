"""
Plots comparing Brazil's local, region-level beta estimates against the
single Brazil row in the main suite's FAOSTAT country-level results -- the
visual counterpart to validation/brazil/compare_beta_animals.py.

(a) TVP: national beta_<year> line + SE band over the 2012-2021 overlap,
    with the regional distribution for that year as a boxplot.
(b) Linear: histogram of regional full-sample betas, with the single
    national full-sample beta marked as a reference line.

Run with working directory = extensification_factors/.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

NATIONAL_TVP_PATH = Path("outputs") / "beta_animals.csv"
NATIONAL_LINEAR_PATH = Path("outputs") / "linear" / "beta_animals_linear.csv"
REGIONAL_TVP_PATH = Path("outputs") / "brazil" / "beta_animals_local_Brazil.csv"
REGIONAL_LINEAR_PATH = Path("outputs") / "brazil" / "beta_animals_linear_local_Brazil.csv"
FIGS_PATH = Path("..") / "figs" / "brazil"
SAVE = True

NATIONAL_COLOR = "#2a78d6"  # blue -- matches BASE_PALETTE[0] used across sibling plotting scripts
REGIONAL_COLOR = "#eb6834"  # orange -- matches BASE_PALETTE[1]

os.makedirs(FIGS_PATH, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

national_tvp = pd.read_csv(NATIONAL_TVP_PATH)
national_linear = pd.read_csv(NATIONAL_LINEAR_PATH)
regional_tvp = pd.read_csv(REGIONAL_TVP_PATH)
regional_linear = pd.read_csv(REGIONAL_LINEAR_PATH)

national_tvp_bra = national_tvp[national_tvp["Area"] == "Brazil"].iloc[0]
national_linear_bra = national_linear[national_linear["Area"] == "Brazil"].iloc[0]

# ---------------------------------------------------------------------------
# (a) TVP time series: national line + SE band vs. regional boxplots
# ---------------------------------------------------------------------------

national_years = {int(c.split("_")[1]) for c in national_tvp.columns if c.startswith("beta_")}
regional_years = {int(c.split("_")[1]) for c in regional_tvp.columns if c.startswith("beta_")}
overlap_years = sorted(national_years & regional_years)

national_beta = np.array([national_tvp_bra[f"beta_{y}"] for y in overlap_years], dtype=float)
national_se = np.array([national_tvp_bra[f"se_{y}"] for y in overlap_years], dtype=float)
regional_by_year = [regional_tvp[f"beta_{y}"].dropna().values for y in overlap_years]

fig, ax = plt.subplots(figsize=(10, 6))

for yline in (0, 1):
    ax.axhline(y=yline, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

bp = ax.boxplot(
    regional_by_year, positions=overlap_years, widths=0.6, patch_artist=True,
    showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.25, markerfacecolor=REGIONAL_COLOR, markeredgecolor="none"),
    medianprops=dict(color=REGIONAL_COLOR, linewidth=1.5),
    whiskerprops=dict(color=REGIONAL_COLOR, alpha=0.6),
    capprops=dict(color=REGIONAL_COLOR, alpha=0.6),
    zorder=2,
)
for patch in bp["boxes"]:
    patch.set_facecolor(REGIONAL_COLOR)
    patch.set_alpha(0.35)
    patch.set_edgecolor(REGIONAL_COLOR)

ax.fill_between(
    overlap_years, national_beta - national_se, national_beta + national_se,
    color=NATIONAL_COLOR, alpha=0.15, linewidth=0, zorder=3,
)
ax.plot(overlap_years, national_beta, color=NATIONAL_COLOR, linewidth=2, marker="o", markersize=4, zorder=4)

ax.set_xticks(overlap_years)
ax.set_xlabel("Year")
ax.set_ylabel(r"Extensification parameter $\beta$ (pasture-based animal products, TVP/Kalman)")
ax.set_title("Brazil: national (FAOSTAT) vs. regional (local pasture data) extensification beta")
ax.legend(
    handles=[
        Line2D([0], [0], color=NATIONAL_COLOR, linewidth=2, marker="o", markersize=4, label="National (FAOSTAT)"),
        Patch(facecolor=REGIONAL_COLOR, alpha=0.35, edgecolor=REGIONAL_COLOR, label=f"Regional (n={len(regional_tvp)} immediate regions)"),
    ],
    loc="best",
)

fig.tight_layout()
if SAVE:
    fig.savefig(FIGS_PATH / "beta_timeseries_comparison_animals_brazil.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# (b) Linear: regional histogram vs. single national reference value
# ---------------------------------------------------------------------------

regional_linear_beta = regional_linear["beta"].dropna().values
national_linear_beta = float(national_linear_bra["beta"])

fig2, ax2 = plt.subplots(figsize=(9, 6))

for xline in (0, 1):
    ax2.axvline(x=xline, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

ax2.hist(regional_linear_beta, bins=30, color=REGIONAL_COLOR, alpha=0.6, edgecolor="white",
          label=f"Regional (n={len(regional_linear_beta)} immediate regions)")
ax2.axvline(national_linear_beta, color=NATIONAL_COLOR, linewidth=2.5,
            label=f"National (FAOSTAT): {national_linear_beta:.3f}")

ax2.set_xlabel(r"Extensification parameter $\beta$ (pasture-based animal products, static OLS, full sample)")
ax2.set_ylabel("Number of regions")
ax2.set_title("Brazil: national (FAOSTAT) vs. regional (local pasture data) extensification beta -- linear model")
ax2.legend(loc="best")

fig2.tight_layout()
if SAVE:
    fig2.savefig(FIGS_PATH / "beta_linear_comparison_animals_brazil.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figures to {FIGS_PATH}")
