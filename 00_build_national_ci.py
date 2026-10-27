"""Build national annual cropping intensity (CI) from the Liu et al. (2021)
Global Cropping Intensity (GCI) rasters (Sci Data, doi:10.1038/s41597-021-01065-9).

GCI pixels hold the number of cropping cycles detected that year (0 = none,
1/2/3 = single/double/triple) on a ~250-m EPSG:4326 grid; the matching QC band
is 0-3 = best/good/fair/poor. GCI was only computed inside the SASAM-GC
cropland extent (cropland > 10%), and a 0 means either "not cropland" or
"cropland with no detected cycle" (fallow/idle) -- the rasters alone can't
tell these apart. So two national CIs are built, each a ratio of
area-weighted sums (EPSG:4326 pixels shrink with cos(latitude), so every
pixel is weighted by its true area):

  ci_cropped  = sum(a * CI) / sum(a * [CI >= 1])
      harvests per unit of land actually cropped that year; GCI only, fallow
      excluded.
  ci_cropland = sum(a * f * CI) / sum(a * f)        over f > 10%
      harvests per unit of cropland incl. fallow, with f the SASAM-GC (~2010,
      static) cropland fraction -- comparable to FAOSTAT harvested area /
      cropland area, which is what Liu et al. validate against.

Countries are geoBoundaries ADM0 polygons (plotting/_geo.load_world),
rasterized onto the GCI grid. Run from extensification_factors/ (needs
0_get_extract_data.py to have run first)."""

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

sys.path.insert(0, str(Path(__file__).resolve().parent / "plotting"))
from _geo import load_country_codes, load_world

CI_DIR: Path = Path("data") / "inputs" / "cropping_intensity"
CROPLAND_PATH: Path = Path("data") / "inputs" / "cropland_extent" / "Global-cropland-percentage-map.tif"
OUT_PATH: Path = Path("data") / "processed" / "national_cropping_intensity.csv"

YEARS: list[int] = list(range(2001, 2020))
CROPLAND_MIN_FRACTION: float = 0.10  # GCI's own processing extent (Liu et al. 2021, Methods); below it a 0 is "not assessed", not fallow
QC_POOR: int = 3
QC_UNREADABLE: int = 255  # a few QC rasters (2002, 2004, 2008, 2016, 2017) have corrupt tiles in the published files
STRIP_ROWS: int = 128  # matches the GCI GeoTIFF block height
N_WORKERS: int = min(8, os.cpu_count() or 1)  # ~1 GB peak RAM per worker
EARTH_AUTHALIC_RADIUS_M: float = 6371007.181  # sphere with the WGS84 ellipsoid's surface area

STATS = ["cropped_ha", "ci_cycles_ha", "cropland_ha", "cropland_cycles_ha", "qc_known_ha", "qc_poor_ha"]


def ci_path(year: int) -> Path:
    return CI_DIR / f"GCI_{year}.tif"


def qc_path(year: int) -> Path:
    return CI_DIR / f"GCI_QC_{year}.tif"


def row_areas_ha(transform, height: int) -> np.ndarray:
    """Area (ha) of one pixel in each row of a north-up EPSG:4326 grid."""
    lat_edges = np.radians(transform.f + transform.e * np.arange(height + 1))
    dlon = np.radians(transform.a)
    return EARTH_AUTHALIC_RADIUS_M ** 2 * dlon * np.abs(np.diff(np.sin(lat_edges))) / 1e4


def read_tolerant(ds, window: Window, fill: int) -> np.ndarray:
    """Read `window`, falling back to block-by-block reads if it hits a
    corrupt tile; unreadable blocks are set to `fill`."""
    try:
        return ds.read(1, window=window)
    except rasterio.errors.RasterioIOError:
        out = np.full((int(window.height), int(window.width)), fill, dtype=ds.dtypes[0])
        bh, bw = ds.block_shapes[0]
        r0, c0 = int(window.row_off), int(window.col_off)
        for r in range(0, out.shape[0], bh):
            for c in range(0, out.shape[1], bw):
                h, w = min(bh, out.shape[0] - r), min(bw, out.shape[1] - c)
                try:
                    out[r:r + h, c:c + w] = ds.read(1, window=Window(c0 + c, r0 + r, w, h))
                except rasterio.errors.RasterioIOError:
                    pass
        return out


# ---------------------------------------------------------------------------
# Per-strip worker. Each process opens its own dataset handles (rasterio
# handles aren't fork-safe) and keeps them for every strip it processes.
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(shapes: list[tuple], n_zones: int) -> None:
    _W["shapes"] = shapes  # (geometry, zone_id, (minx, miny, maxx, maxy))
    _W["n_zones"] = n_zones
    _W["ci"] = {y: rasterio.open(ci_path(y)) for y in YEARS}
    _W["qc"] = {y: rasterio.open(qc_path(y)) for y in YEARS}
    _W["cropland"] = rasterio.open(CROPLAND_PATH)
    ref = _W["ci"][YEARS[0]]
    _W["row_area"] = row_areas_ha(ref.transform, ref.height)


def _process_strip(row_off: int) -> np.ndarray | None:
    """Returns per-zone sums, shape (len(STATS), len(YEARS), n_zones + 1),
    or None if the strip touches no country."""
    ref = _W["ci"][YEARS[0]]
    n_rows = min(STRIP_ROWS, ref.height - row_off)
    window = Window(0, row_off, ref.width, n_rows)
    wt = window_transform(window, ref.transform)
    top, bottom = wt.f, wt.f + wt.e * n_rows

    shapes = [(g, z) for g, z, (_, miny, _, maxy) in _W["shapes"] if miny <= top and maxy >= bottom]
    if not shapes:
        return None
    zones = rasterize(shapes, out_shape=(n_rows, ref.width), transform=wt, fill=0, dtype="uint16").ravel()
    if not zones.any():
        return None

    frac = np.zeros((n_rows, ref.width), dtype="float32")
    reproject(rasterio.band(_W["cropland"], 1), frac, dst_transform=wt, dst_crs=ref.crs,
              dst_nodata=0, resampling=Resampling.nearest)
    frac = frac.ravel()
    frac[~(frac > CROPLAND_MIN_FRACTION)] = 0  # also zeroes any NaN

    row_area = _W["row_area"][row_off:row_off + n_rows]
    n_bins = _W["n_zones"] + 1
    out = np.zeros((len(STATS), len(YEARS), n_bins))

    cl = np.flatnonzero(frac)
    z_cl = zones[cl]
    a_cl = row_area[cl // ref.width] * frac[cl]
    cropland_ha = np.bincount(z_cl, weights=a_cl, minlength=n_bins)

    for t, year in enumerate(YEARS):
        ci = _W["ci"][year].read(1, window=window).ravel()
        idx = np.flatnonzero(ci >= 1)  # CI = 0 adds nothing to any numerator
        z = zones[idx]
        a = row_area[idx // ref.width]
        c = ci[idx].astype("float64")
        # QC = 3 ("poor") only ever occurs with CI = 0, so it's tallied over
        # cropland: poor pixels sit in ci_cropland's denominator as if fallow
        qc = read_tolerant(_W["qc"][year], window, QC_UNREADABLE).ravel()[cl]

        out[0, t] = np.bincount(z, weights=a, minlength=n_bins)
        out[1, t] = np.bincount(z, weights=a * c, minlength=n_bins)
        out[2, t] = cropland_ha
        out[3, t] = np.bincount(z, weights=a * frac[idx] * c, minlength=n_bins)
        out[4, t] = np.bincount(z_cl, weights=a_cl * (qc != QC_UNREADABLE), minlength=n_bins)
        out[5, t] = np.bincount(z_cl, weights=a_cl * (qc == QC_POOR), minlength=n_bins)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    world = load_world().reset_index(drop=True)
    world["zone"] = np.arange(1, len(world) + 1)
    shapes = [(g, int(z), g.bounds) for g, z in zip(world.geometry, world["zone"])]

    with rasterio.open(ci_path(YEARS[0])) as ref:
        height = ref.height
    row_offs = list(range(0, height, STRIP_ROWS))

    totals = np.zeros((len(STATS), len(YEARS), len(world) + 1))
    start = time.time()
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(shapes, len(world))) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_strip, row_offs), 1):
            if res is not None:
                totals += res
            if i % 25 == 0 or i == len(row_offs):
                print(f"{i}/{len(row_offs)} strips, {time.time() - start:.0f}s")

    out = pd.concat([
        pd.DataFrame({"zone": world["zone"].to_numpy(), "Year": year,
                      **{s: totals[k, t, 1:] for k, s in enumerate(STATS)}})
        for t, year in enumerate(YEARS)
    ])
    out = out.merge(world[["zone", "ISO3", "NAME"]], on="zone").drop(columns="zone")

    with np.errstate(invalid="ignore", divide="ignore"):
        out["ci_cropped"] = out["ci_cycles_ha"] / out["cropped_ha"]
        out["ci_cropland"] = out["cropland_cycles_ha"] / out["cropland_ha"]
        out["qc_poor_share"] = out["qc_poor_ha"] / out["qc_known_ha"]  # of cropland area with readable QC

    codes = load_country_codes().rename(columns={"FAOSTAT": "Area Code"})
    out = out.merge(codes, on="ISO3", how="left")
    out = out[["ISO3", "Area Code", "NAME", "Year", "ci_cropped", "ci_cropland",
               "cropped_ha", "cropland_ha", "qc_poor_share"]].sort_values(["ISO3", "Year"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows ({out['ISO3'].nunique()} countries) to {OUT_PATH}")
