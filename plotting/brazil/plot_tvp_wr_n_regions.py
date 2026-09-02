"""
Scatter plots checking whether TVP performance (win rate, MAE) per Brazilian
state depends on how many regions that state contributes -- reads the
by-uf breakdown from validation/brazil/holdout_comparison_animals_brazil.py.

Run with working directory = extensification_factors/.
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BY_UF_PATH = Path("outputs") / "validation" / "brazil" / "holdout_comparison_animals_brazil_by_uf.csv"
FIGS_PATH = Path("..") / "figs" / "brazil"
SAVE = True

POINT_COLOR = "#2a78d6"  # blue -- matches BASE_PALETTE[0] used across sibling plotting scripts

os.makedirs(FIGS_PATH, exist_ok=True)

by_uf = pd.read_csv(BY_UF_PATH)

# ---------------------------------------------------------------------------
# (a) n_regions vs. TVP win rate
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(by_uf["n_regions"], by_uf["weighted_tvp_win_share"], color=POINT_COLOR, alpha=0.7, edgecolor="white")
ax.axhline(y=0.5, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Number of regions (uf)")
ax.set_ylabel("TVP win rate (production-weighted)")
ax.set_title("Brazil: TVP win rate vs. number of regions per state")

fig.tight_layout()
if SAVE:
    fig.savefig(FIGS_PATH / "tvp_winrate_vs_n_regions_brazil.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# (b) n_regions vs. TVP MAE
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(by_uf["n_regions"], by_uf["weighted_mae_tvp"], color=POINT_COLOR, alpha=0.7, edgecolor="white")
ax2.set_xlabel("Number of regions (uf)")
ax2.set_ylabel("TVP MAE (production-weighted)")
ax2.set_title("Brazil: TVP MAE vs. number of regions per state")

fig2.tight_layout()
if SAVE:
    fig2.savefig(FIGS_PATH / "tvp_mae_vs_n_regions_brazil.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figures to {FIGS_PATH}")
