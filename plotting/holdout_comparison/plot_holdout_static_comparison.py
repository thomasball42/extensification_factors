"""
Plots for the TVP vs. static-OLS holdout comparison
(validations_scripts/hold_out_static_comparison.py). Aggregate FAOSTAT
items are already excluded from the input CSVs, so nothing here needs to
filter them out again.

Reads:
  outputs/validation/holdout_static_comparison_pweighted.csv     (per-series)
  outputs/validation/holdout_static_comparison_by_item_pweighted.csv (per-item)

Produces:
  mae_scatter.png                 -- per-series static vs TVP MAE, log-log
  item_mae_comparison.png         -- per-item weighted MAE, static vs TVP
  item_win_share.png              -- per-item weighted TVP win share
  relative_improvement_hist.png   -- signed relative MAE improvement, unweighted vs production-weighted
"""

import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

dat_path = Path("outputs") / "validation" / "holdout_static_comparison_pweighted.csv"
by_item_path = Path("outputs") / "validation" / "holdout_static_comparison_by_item_pweighted.csv"
figs_path = Path("..") / "figs" / "holdout_comparison" / "holdout_static_comparison"

SAVE = True
TOP_N_ITEMS = 15   # by total_production_weight, for the per-item panels

os.makedirs(figs_path, exist_ok=True)

BASE_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
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


COLOR_WIN = BASE_PALETTE[2]   # aqua -- TVP beats static
COLOR_LOSE = BASE_PALETTE[7]  # red  -- static beats TVP

df = pd.read_csv(dat_path)
by_item = pd.read_csv(by_item_path)

# ---------------------------------------------------------------------------
# 1. Per-series scatter: static MAE vs TVP MAE (log-log)
# ---------------------------------------------------------------------------

fig1, ax1 = plt.subplots(figsize=(7, 7))

sizes = np.log10(df["test_avg_production"].clip(lower=1))
sizes = 8 + 60 * (sizes - sizes.min()) / (sizes.max() - sizes.min())

win_mask = df["tvp_beats_static"]
for mask, color, label in [(win_mask, COLOR_WIN, "TVP wins"), (~win_mask, COLOR_LOSE, "Static wins")]:
    ax1.scatter(
        df.loc[mask, "mae_static"], df.loc[mask, "mae_tvp"],
        s=sizes[mask], c=color, alpha=0.35, edgecolor="none", label=label,
    )

positive_mae = pd.concat([df["mae_static"], df["mae_tvp"]])
positive_mae = positive_mae[positive_mae > 0]
lims = [
    positive_mae.min() * 0.8,
    positive_mae.max() * 1.2,
]
ax1.plot(lims, lims, color="0.3", linestyle="--", linewidth=1, zorder=0, label="y = x")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_xlabel("Static-OLS held-out MAE (log points)")
ax1.set_ylabel("TVP held-out MAE (log points)")
ax1.set_title(f"Held-out prediction error, static vs. TVP ({len(df)} series)\n"
              f"TVP wins {win_mask.mean():.1%} of series (point size ~ test-period production)")
ax1.legend(loc="upper left", framealpha=0.9)
fig1.tight_layout()

# ---------------------------------------------------------------------------
# 2. Per-item weighted MAE comparison (top items by production weight)
# ---------------------------------------------------------------------------

top_items = by_item.sort_values("total_production_weight", ascending=False).head(TOP_N_ITEMS)
top_items = top_items.iloc[::-1]  # largest weight at top of horizontal bar chart

fig2, ax2 = plt.subplots(figsize=(8, 0.4 * len(top_items) + 2))
y_pos = np.arange(len(top_items))
bar_h = 0.35

ax2.barh(y_pos + bar_h / 2, top_items["weighted_mae_static"], height=bar_h,
         color=BASE_PALETTE[0], label="Static baseline")
ax2.barh(y_pos - bar_h / 2, top_items["weighted_mae_tvp"], height=bar_h,
         color=BASE_PALETTE[1], label="TVP")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_items["Item"])
ax2.set_xlabel("Production-weighted held-out MAE (log points)")
ax2.set_title(f"Top {len(top_items)} items by held-out test-period production")
ax2.legend(loc="lower right")
fig2.tight_layout()

# ---------------------------------------------------------------------------
# 3. Per-item TVP win share (same items/order as fig 2)
# ---------------------------------------------------------------------------

fig3, ax3 = plt.subplots(figsize=(8, 0.4 * len(top_items) + 2))
win_share = top_items["weighted_tvp_win_share"]
bar_colors = [COLOR_WIN if w >= 0.5 else COLOR_LOSE for w in win_share]

ax3.barh(y_pos, win_share, color=bar_colors, alpha=0.85)
ax3.axvline(0.5, color="0.3", linestyle="--", linewidth=1)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_items["Item"])
ax3.set_xlim(0, 1)
ax3.set_xlabel("Production-weighted share of series where TVP beats static")
ax3.set_title(f"TVP win share by item (top {len(top_items)} by production)")
fig3.tight_layout()

# ---------------------------------------------------------------------------
# 4. Distribution of signed relative improvement, unweighted vs weighted
# ---------------------------------------------------------------------------

rel_imp = (df["mae_static"] - df["mae_tvp"]) / df["mae_static"]
rel_imp_clipped = rel_imp.clip(-1, 1)  # a handful of series have extreme ratios; clip for a readable histogram
weights = df["test_avg_production"].values

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

ax4a.hist(rel_imp_clipped, bins=60, color=BASE_PALETTE[0], alpha=0.8)
ax4a.axvline(0, color="0.3", linestyle="--", linewidth=1)
ax4a.axvline(rel_imp.mean(), color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
             label=f"mean {rel_imp.mean():.1%}")
ax4a.axvline(rel_imp.median(), color=BASE_PALETTE[5], linestyle="-", linewidth=1.5,
             label=f"median {rel_imp.median():.1%}")
ax4a.set_xlabel("Relative MAE improvement, TVP vs. static\n(positive = TVP better)")
ax4a.set_ylabel("Number of series")
ax4a.set_title("Unweighted (every series counts equally)")
ax4a.legend()

weighted_mean = np.average(rel_imp_clipped, weights=weights)
ax4b.hist(rel_imp_clipped, bins=60, weights=weights, color=BASE_PALETTE[2], alpha=0.8)
ax4b.axvline(0, color="0.3", linestyle="--", linewidth=1)
ax4b.axvline(weighted_mean, color=BASE_PALETTE[7], linestyle="-", linewidth=1.5,
             label=f"weighted mean {weighted_mean:.1%}")
ax4b.set_xlabel("Relative MAE improvement, TVP vs. static\n(positive = TVP better)")
ax4b.set_ylabel("Production (test-period, summed)")
ax4b.set_title("Production-weighted")
ax4b.legend()

fig4.suptitle("Distribution of TVP's relative improvement over the static baseline")
fig4.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "mae_scatter.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "item_mae_comparison.png", dpi=300, bbox_inches="tight")
    fig3.savefig(figs_path / "item_win_share.png", dpi=300, bbox_inches="tight")
    fig4.savefig(figs_path / "relative_improvement_hist.png", dpi=300, bbox_inches="tight")
