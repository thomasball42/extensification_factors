"""
Plots for the physical- vs harvested-area comparison
(validation/area_method_comparison.py). Aggregate FAOSTAT items are excluded.

Reads (outputs/physical_area/validation/):
  area_method_comparison_series.csv, area_method_comparison_by_item.csv

Produces (../figs/validation/physical_area/):
  area_method_scatter.png  -- physical vs harvested beta: published products, and same-window OLS
  area_method_by_item.png  -- per-item split of the harvested beta into window, land and cropping-intensity parts
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
from area_method_comparison import OUT_DIR

figs_path = Path("..") / "figs" / "validation" / "physical_area"

SAVE = True
TOP_N_ITEMS = 15  # by physical area

os.makedirs(figs_path, exist_ok=True)

series = pd.read_csv(OUT_DIR / "area_method_comparison_series.csv")
by_item = pd.read_csv(OUT_DIR / "area_method_comparison_by_item.csv")
area_col = next(c for c in series.columns if c.startswith("current_physical_area_"))
prod_col = next(c for c in series.columns if c.startswith("current_production_"))
series = series[~series["is_aggregate"] & series[prod_col].gt(0)]  # the series area_method_comparison.py summarises

# ---------------------------------------------------------------------------
# 1. Physical vs harvested beta, per series
# ---------------------------------------------------------------------------

panels = [
    ("beta_harv_tvp_same_year", "beta_phys_tvp", "Published products (TVP), same year",
     "Harvested-area β (full sample)", "Physical-area β (2002-2019)"),
    ("beta_h_window", "beta_phys_window", "Same 2002-2019 window (OLS): area definition only",
     "Harvested-area β", "Physical-area β"),
]

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))
for ax, (xc, yc, title, xlabel, ylabel) in zip(axes1, panels):
    d = series[[xc, yc, area_col]].dropna()
    sizes = np.log10(d[area_col].clip(lower=1))
    sizes = 4 + 60 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
    lo, hi = np.percentile(np.concatenate([d[xc], d[yc]]), [1, 99])
    ax.scatter(d[xc], d[yc], s=sizes, c="#2a78d6", alpha=0.3, edgecolor="none")
    ax.plot([lo, hi], [lo, hi], color="0.3", linestyle="--", linewidth=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nr = {d[xc].corr(d[yc]):.2f}, {len(d)} series")
fig1.suptitle("Physical- vs harvested-area β per country-crop series (point size ~ physical area; 1st-99th percentile shown)")
fig1.tight_layout()

# ---------------------------------------------------------------------------
# 2. Per-item split of the harvested beta
# ---------------------------------------------------------------------------

bars = [
    ("beta_h_full", "Harvested area, full sample", "0.6"),
    ("beta_h_window", "Harvested area, 2002-2019", "#eb6834"),
    ("beta_phys_window", "Physical area, 2002-2019 (β_phys)", "#2a78d6"),
    ("beta_ci_window", "Cropping intensity, 2002-2019 (β_CI)", "#1baf7a"),
]
top = by_item.head(TOP_N_ITEMS).iloc[::-1]  # largest area at the top of the chart

fig2, ax2 = plt.subplots(figsize=(9, 0.55 * len(top) + 2))
y_pos = np.arange(len(top))
bar_h = 0.8 / len(bars)
for i, (col, label, color) in enumerate(bars):
    ax2.barh(y_pos + (len(bars) / 2 - i - 0.5) * bar_h, top[col], height=bar_h, color=color, label=label)
ax2.axvline(0, color="0.3", linewidth=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top["Item"])
ax2.set_xlabel("Production-weighted β (OLS); β_harvested = β_phys + β_CI on the same window")
ax2.set_title(f"Top {len(top)} crops by physical area")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
fig2.tight_layout()

plt.show()

if SAVE:
    fig1.savefig(figs_path / "area_method_scatter.png", dpi=300, bbox_inches="tight")
    fig2.savefig(figs_path / "area_method_by_item.png", dpi=300, bbox_inches="tight")
