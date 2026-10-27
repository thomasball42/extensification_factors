"""
Plots for the held-out validation (validation/holdout.py).

Reads (outputs/<area_method>/validation/):
  holdout_series.csv, holdout_summary.csv

Produces (../figs/validation/<area_method>/):
  holdout_land_error.png             -- land-use prediction error by arm and horizon
  holdout_predicted_vs_observed.png  -- predicted vs observed change in area over the held-out years
  holdout_tvp_vs_static.png          -- per-series TVP vs static error, materially different betas highlighted
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
from _utils import AREA_METHOD, VALIDATION_OUTPUTS_PATH
from holdout import BETA_DIFF_THRESHOLD, HOLDOUT_YEARS, OUT_SERIES_PATH, OUT_SUMMARY_PATH

figs_path = Path("..") / "figs" / "validation" / AREA_METHOD

SAVE = True

os.makedirs(figs_path, exist_ok=True)

ARM_STYLE = {  # arm: (label, color)
    "tvp": ("TVP β (published)", "#2a78d6"),
    "static": ("Static OLS β (linear)", "#eb6834"),
    "proportional": ("β = 1 (fixed land per tonne)", "#4a3aa7"),
    "no_change": ("β = 0 (no change)", "0.45"),
}
COLOR_DIFFER = "#e34948"

res = pd.read_csv(OUT_SERIES_PATH)
summary = pd.read_csv(OUT_SUMMARY_PATH)
last = res[res["h"] == HOLDOUT_YEARS]
n_series = len(last)
weights = last["A_obs"]

# ---------------------------------------------------------------------------
# 1. Land-use prediction error by arm and horizon
# ---------------------------------------------------------------------------

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))
for arm, (label, color) in ARM_STYLE.items():
    s = summary[summary["arm"] == arm].sort_values("h")
    ax1a.plot(s["h"], 100 * s["area_weighted_abs_err"], marker="o", color=color, label=label)
    ax1b.plot(s["h"], 100 * s["median_abs_pct_err"], marker="o", color=color, label=label)

no_prior = summary[(summary["arm"] == "tvp_no_prior") & (summary["h"] == HOLDOUT_YEARS)].iloc[0]
ax1a.text(0.02, 0.98, f"TVP without the Q prior: {no_prior['area_weighted_abs_err']:.0%} at {HOLDOUT_YEARS} years",
          transform=ax1a.transAxes, va="top", fontsize=9, color="0.3")
for ax, ylabel in [(ax1a, "Area-weighted error, Σ|Â − A| / ΣA (%)"), (ax1b, "Median |Â − A| / A across series (%)")]:
    ax.set_xticks(range(1, HOLDOUT_YEARS + 1))
    ax.set_xlabel("Years after the end of training")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
ax1b.legend(loc="lower right", framealpha=0.9)
fig1.suptitle(f"Held-out land-use prediction ({AREA_METHOD.replace('_', ' ')}, {n_series} series): "
              f"area predicted from A_T and observed production")
fig1.tight_layout()

# ---------------------------------------------------------------------------
# 2. Predicted vs observed change in area over the held-out years
# ---------------------------------------------------------------------------

observed = np.log(last["A_obs"] / last["A_T"])
predicted = {arm: np.log(last[f"pred_{arm}"] / last["A_T"]) for arm in ["tvp", "proportional"]}
lo, hi = np.nanpercentile(np.concatenate([observed, *predicted.values()]), [1, 99])
sizes = np.log10(weights.clip(lower=1))
sizes = 4 + 60 * (sizes - sizes.min()) / (sizes.max() - sizes.min())

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
for ax, arm in zip(axes2, predicted):
    label, color = ARM_STYLE[arm]
    s = summary[(summary["arm"] == arm) & (summary["h"] == HOLDOUT_YEARS)].iloc[0]
    ax.scatter(observed, predicted[arm], s=sizes, c=color, alpha=0.3, edgecolor="none")
    ax.plot([lo, hi], [lo, hi], color="0.3", linestyle="--", linewidth=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Observed ln(A_T+{HOLDOUT_YEARS} / A_T)")
    title = f"{label}\narea-weighted error {s['area_weighted_abs_err']:.1%}"
    if arm == "tvp":
        title += f", 90% intervals cover {s['cover90_area_weighted']:.0%} of area"
    ax.set_title(title)
axes2[0].set_ylabel(f"Predicted ln(A_T+{HOLDOUT_YEARS} / A_T)")
fig2.suptitle(f"Change in area over the {HOLDOUT_YEARS} held-out years (point size ~ area; 1st-99th percentile shown)")
fig2.tight_layout()

# ---------------------------------------------------------------------------
# 3. Per-series TVP vs static error
# ---------------------------------------------------------------------------

err_tvp = last["err_tvp"].abs().clip(lower=1e-4)
err_static = last["err_static"].abs().clip(lower=1e-4)
differ = (last["beta_tvp"] - last["beta_static"]).abs() > BETA_DIFF_THRESHOLD
w_differ = weights[differ]
tvp_better = (err_tvp < err_static)[differ]

fig3, ax3 = plt.subplots(figsize=(7, 7))
ax3.scatter(err_static[~differ], err_tvp[~differ], s=sizes[~differ], c="0.6", alpha=0.25, edgecolor="none",
            label=f"|β_TVP − β_static| ≤ {BETA_DIFF_THRESHOLD} ({(~differ).mean():.0%} of series)")
ax3.scatter(err_static[differ], err_tvp[differ], s=sizes[differ], c=COLOR_DIFFER, alpha=0.5, edgecolor="none",
            label=f"|β_TVP − β_static| > {BETA_DIFF_THRESHOLD} ({differ.mean():.0%} of series)")
lims = [1e-4, max(err_tvp.max(), err_static.max()) * 1.2]
ax3.plot(lims, lims, color="0.3", linestyle="--", linewidth=1, zorder=0)
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlim(lims)
ax3.set_ylim(lims)
ax3.set_xlabel(f"Static OLS β: |Â − A| / A, {HOLDOUT_YEARS} years on")
ax3.set_ylabel(f"TVP β: |Â − A| / A, {HOLDOUT_YEARS} years on")
ax3.set_title(f"Where the betas differ, TVP is closer for {tvp_better.mean():.0%} of series "
              f"({np.average(tvp_better, weights=w_differ):.0%} of area)")
ax3.legend(loc="upper left", framealpha=0.9)
fig3.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "holdout_land_error.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "holdout_predicted_vs_observed.png", dpi=300, bbox_inches="tight")
    fig3.savefig(figs_path / "holdout_tvp_vs_static.png", dpi=300, bbox_inches="tight")
