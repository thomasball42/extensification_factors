"""
World map of the extensification parameter beta (prior'd TVP/Kalman beta,
see config.json's "q_prior") for pasture-based animal products, at
N_WINDOWS non-overlapping time windows spanning the full beta_animals.csv
year range.

beta_animals.csv has one row per Area already (Item is always "All
pasture-based animal products"), so unlike beta_map_crops.py there is no
cross-item weighting to do.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _geo import load_country_codes, load_world, time_windows

dat_path = Path("outputs") / "beta_animals.csv"
figs_path = Path("..") / "figs" / "beta_map_animals"
SAVE = True

N_WINDOWS = 1
START_YEAR = None  # None = start of available data
END_YEAR = None  # None = end of available data
CMAP = "RdBu_r"  

figsize = (10, 3.8 * N_WINDOWS)
norm_min_max = (-0.1, 0.3)
os.makedirs(figs_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Load + prep
# ---------------------------------------------------------------------------

df = pd.read_csv(dat_path)

country_codes = load_country_codes()
df = df.merge(country_codes, left_on="Area Code", right_on="FAOSTAT", how="inner")

beta_years = sorted(int(c.split("_")[1]) for c in df.columns if c.startswith("beta_"))
windows = time_windows(beta_years, N_WINDOWS, start=START_YEAR, end=END_YEAR)

world = load_world()


world_iso3 = set(world["ISO3"])

records = []
for _, row in df.iterrows():
    iso3 = row["ISO3"]
    area = row["Area"]

    if iso3 not in world_iso3:
        print(f"[no map polygon] {area} ({iso3}) not found in world shapefile")

    rec = {"ISO3": iso3, "Area": area}
    for label, window_years in windows:
        beta_cols = [f"beta_{y}" for y in window_years]
        vals = row[beta_cols].astype(float)
        n_valid = vals.notna().sum()
        mean = vals.mean(skipna=True) if n_valid > 0 else np.nan
        rec[label] = mean

        if n_valid == 0:
            print(f"[no data] {area} ({iso3}) has no valid beta in window {label}")
        else:
            print(f"{area} ({iso3}) window {label}: mean beta = {mean:.4f} ({n_valid}/{len(beta_cols)} valid years)")
    records.append(rec)

country_betas = pd.DataFrame(records).set_index("ISO3")

window_betas = [country_betas[label].dropna().rename("beta") for label, _ in windows]

all_vals = pd.concat(window_betas).values
vlim = np.nanpercentile(np.abs(all_vals), 99)

norm = TwoSlopeNorm(vmin=norm_min_max[0] if norm_min_max[0] is not None else -vlim, 
                    vcenter=0, 
                    vmax=norm_min_max[1] if norm_min_max[1] is not None else vlim)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(N_WINDOWS, 1, figsize=figsize)

if N_WINDOWS == 1:
    axes = [axes]

for ax, (label, _), beta in zip(axes, windows, window_betas):
    world_w = world.merge(beta, on="ISO3", how="left")
    world_w.plot(
        column="beta", ax=ax, cmap=CMAP, norm=norm,
        edgecolor="0.0", linewidth=0.1,
        missing_kwds={"color": "#9c9b9b", "edgecolor": "0.0", "linewidth": 0.1, "label": "no data"},
    )
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-60, 85)  # crop Antarctica
    for spine in ax.spines.values():
        spine.set_visible(False)

# fig.suptitle(r"Extensification parameter $\beta$ — pasture-based animal products"
#              "\n(prior'd TVP/Kalman beta, time-averaged within each window)")

sm = ScalarMappable(norm=norm, cmap=CMAP)
cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)

cbar.ax.set_yticks([norm.vmin, 0, norm.vmax])
cbar.ax.set_yticklabels([f"<{norm.vmin:.2f}", "0", f">{norm.vmax:.2f}"])
cbar.set_label(f"Extensification parameter $\\beta$")

# fig.tight_layout()

if SAVE:
    fig.savefig(figs_path / "beta_map_animals.png", dpi=300, bbox_inches="tight")

plt.show()
