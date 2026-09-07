"""
Animals counterpart to plot_beta_pred_vs_actual_land_use_crops.py -- see
that module's docstring for the full methodology. Plots how well beta
predicts the pasture-area change that actually happened, from the
per-transition CSV produced by
../../validation/compute_beta_pred_vs_actual_land_use_animals.py -- run that
script first.

Unlike crops, there is no per-Item dimension here (a single pooled "All
pasture-based animal products" series per Area/country -- see
beta_scripts/_compute_beta_animals.py), so series = Area alone and there's
no per-item breakdown panel.

Produces:
  land_use_pred_vs_actual_scatter.png            -- predicted vs. actual
                                                      log-change in pasture
                                                      area (all transitions
                                                      pooled), unweighted vs.
                                                      production-weighted
  land_use_pred_vs_actual_error_distribution.png -- 3x2 grid: one row per
                                                      scenario (A/B/C), one
                                                      column per unweighted /
                                                      production-weighted,
                                                      distribution of the
                                                      prediction error (actual
                                                      - predicted, as % of
                                                      actual area)
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

TRANSITIONS_PATH = Path("outputs") / "land_use_impact" / "land_use_pred_vs_actual_transitions_animals.csv"
figs_path = Path("..") / "figs" / "land_use" / "beta_pred_vs_actual_land_use_animals"
SAVE = True

BASE_PALETTE = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
COLOR_CORRECT = "#1baf7a"
COLOR_WRONG = "#e34948"
SCENARIO_KEYS = ["a", "b", "c"]

os.makedirs(figs_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the per-transition CSV written by
# compute_beta_pred_vs_actual_land_use_animals.py
# ---------------------------------------------------------------------------

trans = pd.read_csv(TRANSITIONS_PATH)
SCENARIO_LABELS = dict(trans[["scenario", "scenario_label"]].drop_duplicates().to_numpy())

n_series = trans[["Area", "Area Code"]].drop_duplicates().shape[0]
print(f"{len(trans)} transitions across {n_series} areas loaded from {TRANSITIONS_PATH.name}.")

actual_a = trans["actual_a"].to_numpy(float)
predicted_a = trans["predicted_a"].to_numpy(float)
correct_direction = trans["correct_direction"].to_numpy(bool)
weight = trans["Production"].to_numpy(float)

r2 = np.corrcoef(actual_a, predicted_a)[0, 1] ** 2
dir_acc = correct_direction.mean()
dir_acc_w = np.average(correct_direction, weights=weight)
print(f"R-squared (predicted vs. actual log-change in pasture area): {r2:.3f}")
print(f"Directional accuracy (unweighted): {dir_acc:.1%}")
print(f"Directional accuracy (production-weighted): {dir_acc_w:.1%}")

# ---------------------------------------------------------------------------
# 1. Predicted vs. actual scatter (log-change in pasture area), all
#    transitions pooled, unweighted vs. production-weighted
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
    ax.set_xlabel("Actual Delta-log(pasture area)")
    ax.set_ylabel("Beta-predicted Delta-log(pasture area)")
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

plt.show()

if SAVE:
    fig1.savefig(figs_path / "land_use_pred_vs_actual_scatter.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "land_use_pred_vs_actual_error_distribution.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved figures to {figs_path}")
