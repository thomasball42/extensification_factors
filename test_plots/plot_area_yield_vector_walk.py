"""
Year-over-year area/yield decomposition, drawn as a connected vector walk.

Each year-over-year step is a vector whose DIRECTION comes from the
(%D Area harvested, %D Yield) split for that year (exactly as in
plot_area_yield_decomposition.py — 45 degrees = equal contribution, along the
x-axis = pure area-driven, along the y-axis = pure yield-driven) but whose
MAGNITUDE is rescaled to |%D Production| for that year, rather than the raw
Euclidean length of (d_area, d_yield). Steps are chained tip-to-tail across
years (starting from an arbitrary origin) so the walk's bearing at each step
shows *how* production changed that year and its length shows *how much*.

Since direction alone can't distinguish a year of growth from a year of
decline (the same area/yield mix can net positive or negative), segments for
years where production fell are drawn in a lightened tint of the area's
color, so declines read as visually "fainter" steps in the walk.

Multiple areas can be overlaid on the same panel; the panel is scoped to a
single commodity (set via ifilt/iexcl, must resolve to exactly one Item).
"""

import colorsys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
import os

DATA_PATH = Path("data") / "inputs"
figs_path = Path("..") / "figs" / "area_yield_vector_walk"

SAVE = True

# filtering

ifilt = [
        "Cereal"
        ]

iexcl = [
        "buckwheat",
        "n.e.c."
        ]

afilt = [
        "Africa",
        "Europe",
        "Northern America",
        "Eastern Asia",
        "Southern Asia",
        "South America",
        ]

aexcl = [
        "taiwan",
        "southern",
        "northern",
        "western",
        "eastern",
        "republic",
        "union",
        "middle",
        "South Africa",
        "central"
        ]

# main
os.makedirs(figs_path, exist_ok=True)

elements = ["Area harvested", "Production", "Yield"]
columns = ["Area", "Area Code", "Item", "Item Code", "Element", "Year", "Value", "Unit"]

df = pd.read_csv(
    DATA_PATH / "Production_Crops_Livestock_E_All_Data_(Normalized).csv",
    encoding="latin-1",
    low_memory=False,
    usecols=columns,
)

df = df.drop(columns=["Unit"])
df = df[df.Element.isin(elements)]

# restrict to items that actually report "Area harvested" (i.e. crops, not
# livestock/animal-product items that reuse the "Production" element label)
crop_items = df.loc[df.Element == "Area harvested", "Item"].unique()
df = df[df.Item.isin(crop_items)]

all_items = df.Item.unique()
all_areas = df.Area.unique()


def _filter_list(all_, filt_, excl_):
    if not filt_:
        return all_
    filt = [f.lower() for f in filt_]
    excl = [f.lower() for f in excl_]
    return [
        i for i in all_
        if any(f in i.lower() for f in filt)
        and (i.lower() in filt or not any(f in i.lower() for f in excl))
    ]


item_matches = _filter_list(all_items, ifilt, iexcl)
assert len(item_matches) == 1, (
    f"ifilt/iexcl must resolve to exactly one Item, got {item_matches}"
)
item = item_matches[0]

areas = _filter_list(all_areas, afilt, aexcl)
assert areas, f"afilt/aexcl matched no Area, candidates were {list(all_areas)[:20]}..."

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


def _light_tint(hex_color, frac=0.75):
    """Blend hex_color toward white by frac (0=color, 1=white) — used to mark
    decline (%D Production < 0) segments as visually "fainter" steps."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


base_colors = dict(zip(areas, distinct_colors(len(areas))))

fig, ax = plt.subplots(figsize=(9, 8))

all_x, all_y = [], []

for area in areas:
    g = df[(df.Item == item) & (df.Area == area)].pivot_table(
        index="Year", columns="Element", values="Value"
    ).sort_index()
    if g.empty:
        continue

    full_years = np.arange(g.index.min(), g.index.max() + 1)
    g = g.reindex(full_years)

    with np.errstate(divide="ignore"):
        log_area = np.log(g["Area harvested"].values)
        log_yield = np.log(g["Yield"].values)

    d_area = 100 * np.diff(log_area)
    d_yield = 100 * np.diff(log_yield)
    years = full_years[1:]

    valid = np.isfinite(d_area) & np.isfinite(d_yield)
    d_area, d_yield, years = d_area[valid], d_yield[valid], years[valid]
    if len(d_area) == 0:
        continue

    d_prod = d_area + d_yield  # exactly additive by construction in log-space

    # direction from the (d_area, d_yield) split; magnitude rescaled to
    # |%D Production| instead of the raw norm of (d_area, d_yield)
    step_norm = np.hypot(d_area, d_yield)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_x = np.where(step_norm > 1e-9, d_area / step_norm, 0.0)
        unit_y = np.where(step_norm > 1e-9, d_yield / step_norm, 0.0)
    vec_x = unit_x * np.abs(d_prod)
    vec_y = unit_y * np.abs(d_prod)

    cum_x = np.concatenate([[0.0], np.cumsum(vec_x)])
    cum_y = np.concatenate([[0.0], np.cumsum(vec_y)])
    all_x.append(cum_x)
    all_y.append(cum_y)

    base_color = base_colors[area]
    grow_color = base_color
    decline_color = _light_tint(base_color)

    for i in range(len(d_prod)):
        seg_color = grow_color if d_prod[i] >= 0 else decline_color
        arrow = FancyArrowPatch(
            (cum_x[i], cum_y[i]), (cum_x[i + 1], cum_y[i + 1]),
            arrowstyle="-|>", mutation_scale=10, color=seg_color,
            linewidth=1.3, zorder=2, shrinkA=0, shrinkB=0,
        )
        ax.add_patch(arrow)

    ax.annotate(str(years[0]), (cum_x[0], cum_y[0]), fontsize=7, color=base_color,
                xytext=(4, 4), textcoords="offset points")
    ax.annotate(str(years[-1]), (cum_x[-1], cum_y[-1]), fontsize=7, color=base_color,
                xytext=(4, 4), textcoords="offset points")

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
lo = min(all_x.min(), all_y.min())
hi = max(all_x.max(), all_y.max())
pad = 0.1 * (hi - lo)
lo, hi = lo - pad, hi + pad

ax.axhline(0, color="0.7", linestyle="--", linewidth=0.8, zorder=0)
ax.axvline(0, color="0.7", linestyle="--", linewidth=0.8, zorder=0)
ax.plot([lo, hi], [lo, hi], color="0.4", linewidth=1, zorder=0)
ax.annotate("equal contribution", (hi, hi), fontsize=8, color="0.4",
            ha="right", va="bottom", rotation=45, rotation_mode="anchor")

ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")

ax.set_xlabel("Cumulative area-directed step (bearing set by %$\\Delta$ Area harvested, "
              "length set by |%$\\Delta$ Production|)")
ax.set_ylabel("Cumulative yield-directed step (bearing set by %$\\Delta$ Yield, "
              "length set by |%$\\Delta$ Production|)")
ax.set_title(f"{item}: area/yield-directed production-change walk")

legend_handles = [Line2D([0], [0], color=c, lw=2) for c in base_colors.values()]
legend_handles += [
    Line2D([0], [0], color="0.3", lw=2, label="production grew that year"),
    Line2D([0], [0], color=_light_tint("#808080"), lw=2, label="production fell that year"),
]
ax.legend(handles=legend_handles, labels=list(base_colors.keys()) + ["growth year", "decline year"],
          title="Area", loc="upper left", framealpha=0.9, fontsize=8)

fig.tight_layout()
plt.show()

if SAVE:
    filename = f"area_yield_vector_walk_{item}_{'_'.join(areas)}.png"
    fig.savefig(figs_path / filename, dpi=300, bbox_inches="tight")
