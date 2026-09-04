import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

GBOUNDARIES_ADM0_API = "https://www.geoboundaries.org/api/current/gbOpen/ALL/ADM0/"
WORLD_DIR = Path("data") / "inputs" / "country_data"
WORLD_META_CACHE = WORLD_DIR / "geoboundaries_adm0_metadata.json"

COUNTRY_CODES_PATH = Path("data") / "inputs" / "nocsDataExport_20251021-164754.xlsx"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def _adm0_metadata() -> list[dict]:
    """Per-country ADM0 metadata (name, ISO3, continent, download links)
    from the geoBoundaries API, cached locally after the first fetch.

    The bulk ALL/ADM0 listing omits India (confirmed against the live API,
    not a caching artefact) even though India has its own ADM0 entry, so
    it's fetched individually and backfilled."""
    if not WORLD_META_CACHE.exists():
        WORLD_META_CACHE.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(GBOUNDARIES_ADM0_API)
        r.raise_for_status()
        entries = r.json()
        if not any(e["boundaryISO"] == "IND" for e in entries):
            r_ind = requests.get("https://www.geoboundaries.org/api/current/gbOpen/IND/ADM0/")
            r_ind.raise_for_status()
            entries.append(r_ind.json())
        WORLD_META_CACHE.write_text(json.dumps(entries), encoding="utf-8")
    return json.loads(WORLD_META_CACHE.read_text(encoding="utf-8"))


def load_world() -> gpd.GeoDataFrame:
    """geoBoundaries simplified ADM0 country polygons, one small GeoJSON
    per country, downloaded once each and cached locally."""
    countries = []
    for entry in _adm0_metadata():
        dest = WORLD_DIR / Path(entry["simplifiedGeometryGeoJSON"]).name
        if not dest.exists():
            _download(entry["simplifiedGeometryGeoJSON"], dest)
        country = gpd.read_file(dest)
        country["ISO3"] = entry["boundaryISO"]
        country["NAME"] = entry["boundaryName"]
        country["CONTINENT"] = entry["Continent"]
        countries.append(country)

    world = pd.concat(countries, ignore_index=True)
    return gpd.GeoDataFrame(world[["ISO3", "CONTINENT", "NAME", "geometry"]], crs=countries[0].crs)


def load_country_codes() -> pd.DataFrame:
    """FAOSTAT "Area Code" -> ISO3 lookup. Left-merging onto this (and
    dropping unmatched rows) also has the effect of dropping FAOSTAT's own
    regional/continental aggregate Areas (e.g. "Europe", "World"), since
    those Area Codes aren't real countries and so aren't in this list."""
    codes = pd.read_excel(COUNTRY_CODES_PATH)
    return codes[["FAOSTAT", "ISO3"]].dropna()


def weighted_mean_se(values, ses, weights):
    """Weighted mean of `values` and its propagated SE, assuming
    independent errors: se = sqrt(sum(w_i^2 se_i^2)) / sum(w_i).
    NaNs (in values or weights) are dropped; a NaN se is treated as 0 (i.e.
    that item contributes to the mean but not to the uncertainty) so one
    missing SE doesn't blank out an otherwise-usable aggregate."""
    values = np.asarray(values, dtype=float)
    ses = np.asarray(ses, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if valid.sum() == 0:
        return np.nan, np.nan
    v, s, w = values[valid], np.nan_to_num(ses[valid], nan=0.0), weights[valid]
    wsum = w.sum()
    mean = float((v * w).sum() / wsum)
    se = float(np.sqrt((w ** 2 * s ** 2).sum()) / wsum)
    return mean, se


def time_windows(years, n=3, start=None, end=None):
    """Split [start, end] (defaults to the full range of `years`) into n
    contiguous, near-equal calendar-year windows, then intersect each with
    the years actually present in `years`. Returns a list of (label,
    year_array) tuples; labels reflect the calendar window bounds even if
    some of those years have no data."""
    years = np.asarray(sorted(years))
    if start is None:
        start = int(years.min())
    if end is None:
        end = int(years.max())
    chunks = np.array_split(np.arange(start, end + 1), n)
    return [
        (f"{c.min()}–{c.max()}", years[(years >= c.min()) & (years <= c.max())])
        for c in chunks
    ]
