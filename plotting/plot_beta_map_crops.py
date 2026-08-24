"""
World map of the extensification parameter beta (prior'd TVP/Kalman beta,
see config.json's "q_prior"), production-weighted across crop items.

The full beta_crops.csv year range is split into N_WINDOWS non-overlapping
windows; the two panels show (1) the most recent window (the "last N
years") and (2) the change between the two earlier, historical windows.

FAOSTAT aggregate items (e.g. "Cereals, primary") are excluded so a
country's per-item betas aren't double counted; each remaining item's
contribution to its country's value is weighted by current production
(same convention as validation/hold_out_static_comparison.py's production
weighting).
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _geo import load_country_codes, load_world, time_windows
from _functions import filter_list as _filter_list

dat_path = Path("outputs") / "beta_crops.csv"
figs_path = Path("..") / "figs" / "beta_map_crops"
SAVE = True

norm_min_max = (None, None)
ZERO_FRAC = 0.1  # fraction of the colorbar's height where 0 sits
N_WINDOWS = 2
START_YEAR = None  # e.g. 2000; None = start of available data
END_YEAR = None  # e.g. 2023; None = end of available data
CMAP = "RdBu_r"  # diverging: blue below 0, red above
CMAP_DIFF = "RdBu_r"  # red = beta rose between historical windows, blue = beta fell

# crop filtering (kept in sync with beta_timeseries_crops.py / beta_linear_crops.py)
ifilt = [
        # "rice",
        "wheat",
        # "maize",
        # "sorghum",
        # "soy",
        # "barley",
        # "Cereal"
        ]

iexcl = [
        "buckwheat", "green corn", "n.e.c."
         ]


figsize = (8, 6)
os.makedirs(figs_path, exist_ok=True)

class ZeroFractionNorm(Normalize):
    """Diverging norm like TwoSlopeNorm, but 0 sits at an arbitrary
    fraction of the colorbar (`zero_frac`) instead of always the midpoint.
    Lets a mostly-positive-valued map keep only a small sliver of the bar
    for the few negative values instead of wasting half of it.

    Degrades gracefully when the data doesn't straddle 0 (e.g. a crop
    filter whose window happens to be all-positive): 0 is then pinned to
    whichever end of the bar is adjacent to the data, rather than crashing."""

    def __init__(self, vmin, vmax, zero_frac=0.25, clip=False):
        super().__init__(vmin, vmax, clip)
        self.zero_frac = zero_frac

    def _breakpoints(self):
        vmin, vmax = self.vmin, self.vmax
        if vmin < 0 < vmax:
            return [vmin, 0, vmax], [0, self.zero_frac, 1]
        if vmin >= 0:  # no negative values: 0 pinned at the bottom
            return [0, vmax], [0, 1]
        return [vmin, 0], [0, 1]  # vmax <= 0: 0 pinned at the top

    def __call__(self, value, clip=None):
        x, y = self._breakpoints()
        return np.ma.masked_array(np.interp(value, x, y, left=y[0], right=y[-1]))

    def inverse(self, value):
        x, y = self._breakpoints()
        return np.interp(value, y, x, left=x[0], right=x[-1])


def zero_fraction_cmap(cmap, vmin, vmax, zero_frac, n=256):
    """Warps a diverging colormap so its center color (assumed to sit at
    index 0.5, e.g. white in RdBu_r) lands at `zero_frac` instead. Needed
    because matplotlib draws a colorbar's gradient by sampling the cmap
    uniformly and only uses the norm to place tick labels — so pairing
    ZeroFractionNorm with an unwarped cmap moves the "0" tick but leaves
    the white center sitting at the old midpoint.

    Mirrors ZeroFractionNorm's fallback: if the data doesn't straddle 0,
    only the relevant half of the diverging cmap (white to the saturated
    color) is used, so a one-signed range doesn't paint values on the
    wrong side of white."""
    base = plt.get_cmap(cmap, n)
    if vmin < 0 < vmax:
        n_lo = min(max(int(round(n * zero_frac)), 1), n - 1)
        n_hi = n - n_lo
        lo = base(np.linspace(0, 0.5, n_lo))
        hi = base(np.linspace(0.5, 1, n_hi))
        return ListedColormap(np.vstack([lo, hi]))
    if vmin >= 0:
        return ListedColormap(base(np.linspace(0.5, 1, n)))
    return ListedColormap(base(np.linspace(0, 0.5, n)))


# ---------------------------------------------------------------------------
# Load + prep
# ---------------------------------------------------------------------------

df = pd.read_csv(dat_path)
df = df[~df["is_aggregate"]].copy()

items = _filter_list(df["Item"].unique(), ifilt, iexcl)
df = df[df["Item"].isin(items)].copy()

prod_col = next(c for c in df.columns if c.startswith("current_production_"))

country_codes = load_country_codes()
df = df.merge(country_codes, left_on="Area Code", right_on="FAOSTAT", how="inner")

beta_years = sorted(int(c.split("_")[1]) for c in df.columns if c.startswith("beta_"))
windows = time_windows(beta_years, N_WINDOWS, start=START_YEAR, end=END_YEAR)

world = load_world()


def window_country_beta(window_years):
    """Production-weighted mean beta per ISO3 country, averaging each
    item's beta_t over the window's years before weighting across items."""
    beta_cols = [f"beta_{y}" for y in window_years]
    item_beta = df[beta_cols].mean(axis=1, skipna=True)
    tmp = pd.DataFrame({"ISO3": df["ISO3"], "beta": item_beta, "weight": df[prod_col]})
    tmp = tmp.dropna(subset=["beta", "weight"])
    tmp = tmp[tmp["weight"] > 0]
    return tmp.groupby("ISO3").apply(
        lambda g: np.average(g["beta"], weights=g["weight"]), include_groups=False
    ).rename("beta")


window_betas = [window_country_beta(years) for _, years in windows]

recent_label, recent_beta = windows[-1][0], window_betas[-1]
hist1_label, hist1_beta = windows[0][0], window_betas[0]
hist2_label, hist2_beta = windows[1][0], window_betas[1]
diff_beta = (hist2_beta - hist1_beta).dropna()
diff_label = f"{hist2_label} minus {hist1_label}"


vmin = norm_min_max[0] if norm_min_max[0] is not None else np.nanmin(recent_beta.values)
vmax = norm_min_max[1] if norm_min_max[1] is not None else np.nanpercentile(recent_beta.values, 99)
norm = ZeroFractionNorm(vmin=vmin, vmax=vmax, zero_frac=ZERO_FRAC)
cmap = zero_fraction_cmap(CMAP, vmin, vmax, ZERO_FRAC)

diff_vlim = np.nanpercentile(np.abs(diff_beta.values), 99)
diff_norm = ZeroFractionNorm(vmin=-diff_vlim, vmax=diff_vlim, zero_frac=0.5)
diff_cmap = zero_fraction_cmap(CMAP_DIFF, -diff_vlim, diff_vlim, 0.5)  # symmetric: no-op warp

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=figsize)

panels = [
    (axes[0], recent_label, recent_beta, cmap, norm),
    (axes[1], diff_label, diff_beta, diff_cmap, diff_norm),
]

for ax, label, beta, cmap, panel_norm in panels:
    world_w = world.merge(beta.rename("beta"), on="ISO3", how="left")
    world_w.plot(
        column="beta", ax=ax, cmap=cmap, norm=panel_norm,
        edgecolor="0.4", linewidth=0.2,
        missing_kwds={"color": "#dddddd", "edgecolor": "0.6", "linewidth": 0.2, "label": "no data"},
    )
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-60, 85)  # crop Antarctica
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = ScalarMappable(norm=panel_norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.05, pad=0.05, shrink=0.8)
    cbar.ax.set_yticks([panel_norm.vmin, 0, panel_norm.vmax])
    cbar.ax.set_yticklabels([f"<{panel_norm.vmin:.2f}", "0", f">{panel_norm.vmax:.2f}"])

# fig.suptitle(r"Extensification parameter $\beta$ — production-weighted mean across crops"
#              "\n(prior'd TVP/Kalman beta, time-averaged within each window)")

fig.tight_layout()

if SAVE:
    fig.savefig(figs_path / "beta_map_crops.png", dpi=300, bbox_inches="tight")

plt.show()
