"""
Estimated vs. actual scatter for Brazil's held-out region-years -- the raw
(actual, predicted) pairs that holdout_comparison_animals_brazil.py's
mae_static/mae_tvp are averaged over. Reads
validation/brazil/holdout_estimates_vs_actual_brazil.py's output.

Points on the y=x line are perfect one-step-ahead predictions; the average
vertical (or horizontal) scatter off that line is exactly the MAE.

Run with working directory = extensification_factors/.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("outputs") / "validation" / "brazil" / "holdout_estimates_vs_actual_brazil.csv"
FIGS_PATH = Path("..") / "figs" / "brazil"
SAVE = True

STATIC_COLOR = "#2a78d6"  # blue -- matches BASE_PALETTE[0] used across sibling plotting scripts
TVP_COLOR = "#eb6834"     # orange -- matches BASE_PALETTE[1]

os.makedirs(FIGS_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)

mae_static = (df["actual"] - df["estimated_static"]).abs().mean()
mae_tvp = (df["actual"] - df["estimated_tvp"]).abs().mean()

lo = min(df["actual"].min(), df["estimated_static"].min(), df["estimated_tvp"].min())
hi = max(df["actual"].max(), df["estimated_static"].max(), df["estimated_tvp"].max())
pad = 0.05 * (hi - lo)
lims = (lo - pad, hi + pad)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

for ax, col, color, label, mae in (
    (ax1, "estimated_static", STATIC_COLOR, "Static OLS", mae_static),
    (ax2, "estimated_tvp", TVP_COLOR, "TVP (Kalman)", mae_tvp),
):
    ax.plot(lims, lims, color="k", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.scatter(df["actual"], df[col], s=14, color=color, alpha=0.45, edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual (demeaned log-area diff)")
    ax.set_title(f"{label}\nMAE = {mae:.4f}  (n = {len(df)})")

ax1.set_ylabel("Estimated (one-step-ahead)")
fig.suptitle("Brazil holdout: estimated vs. actual, held-out region-years")

fig.tight_layout()
if SAVE:
    fig.savefig(FIGS_PATH / "holdout_estimated_vs_actual_brazil.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to {FIGS_PATH}")
