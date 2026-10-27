"""
Plot for the Production = Area harvested * Yield identity check
(validation/production_ay_identity.py): distribution of the relative
discrepancy between FAOSTAT's reported production and area * yield.

Reads:
  outputs/<area_method>/validation/ay_identity_histogram.csv

Produces (../figs/validation/<area_method>/):
  ay_identity_histogram.png
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
from _utils import AREA_METHOD
from production_ay_identity import HIST_CLIP, OUT_HIST_PATH, TOLERANCE

figs_path = Path("..") / "figs" / "validation" / AREA_METHOD

SAVE = True

os.makedirs(figs_path, exist_ok=True)

hist = pd.read_csv(OUT_HIST_PATH)
n_rows = hist["count"].sum()
within_tol = hist.loc[hist["bin_right"] <= TOLERANCE, "count"].sum() / n_rows

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(100 * hist["bin_left"], hist["count"], width=100 * (hist["bin_right"] - hist["bin_left"]),
       align="edge", color="#2a78d6", edgecolor="white", linewidth=0.5)
ax.axvline(100 * TOLERANCE, color="0.3", linestyle="--", linewidth=1)
ax.set_yscale("log")
ax.set_xlim(0, 100 * HIST_CLIP)
ax.set_xlabel(f"|Production − Area harvested × Yield| / Production (%), ≥ {HIST_CLIP:.0%} in the last bin")
ax.set_ylabel("Country-crop-years")
ax.set_title(f"FAOSTAT identity check, {n_rows:,} country-crop-years: "
             f"{within_tol:.2%} within {TOLERANCE:.0%}")
fig.tight_layout()

plt.show()

if SAVE:
    fig.savefig(figs_path / "ay_identity_histogram.png", dpi=300, bbox_inches="tight")
