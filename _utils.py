import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Item/Area name filtering (plotting scripts)
# ---------------------------------------------------------------------------

def filter_list(all_, filt_, excl_):
    """Substring-filter `all_` (case-insensitive) down to entries matching
    any of `filt_`, then drop any of those that also match `excl_` --
    unless the entry is an exact (lowercased) match for one of `filt_`
    entries, which bypasses exclusion. `filt_` empty means "no filtering",
    returning `all_` unchanged (exclusions don't apply in that case)."""
    if not filt_:
        return all_
    filt = [f.lower() for f in filt_]
    excl = [f.lower() for f in excl_]
    return [
        i for i in all_
        if any(f in i.lower() for f in filt)
        and (i.lower() in filt or not any(f in i.lower() for f in excl))
    ]


# ---------------------------------------------------------------------------
# FAOSTAT aggregate-item flagging
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_q_prior_config(config_path: Path = CONFIG_PATH):
    """Returns (q_prior_scale, q_prior_strength) from the "q_prior" section of
    config.json, for use as fit_tvp_beta's shrinkage-prior kwargs."""
    with open(config_path) as f:
        config = json.load(f)

    q_prior_cfg = config.get("q_prior", {})
    return q_prior_cfg.get("q_prior_scale"), q_prior_cfg.get("q_prior_strength", 1.0)


def load_aggregate_matcher(config_path: Path = CONFIG_PATH):
    """Returns an is_aggregate(item_name) -> bool function built from the
    "faostat_aggregates" section of config.json: an exact-match set of known
    aggregate names plus a set of case-insensitive substring patterns."""
    with open(config_path) as f:
        config = json.load(f)

    agg_cfg = config.get("faostat_aggregates", {})
    known_names = set(agg_cfg.get("known_names", []))
    name_patterns = [p.lower() for p in agg_cfg.get("name_patterns", [])]

    def is_aggregate(item_name: str) -> bool:
        if item_name in known_names:
            return True
        item_lower = item_name.lower()
        return any(pattern in item_lower for pattern in name_patterns)

    return is_aggregate


# ---------------------------------------------------------------------------
# plot_config.json (plotting-script filter presets / per-script parameters)
# ---------------------------------------------------------------------------

PLOT_CONFIG_PATH = Path(__file__).parent / "plot_config.json"


def load_plot_config(config_path: Path = PLOT_CONFIG_PATH) -> dict:
    """Returns the full parsed plot_config.json."""
    with open(config_path) as f:
        return json.load(f)


def resolve_filter_preset(plot_config: dict, script_name: str) -> tuple:
    """Looks up plot_config["scripts"][script_name]["active"], resolves it
    against plot_config["filter_presets"], and returns
    (ifilt, iexcl, afilt, aexcl) -- any field the preset doesn't define
    (e.g. a preset with no afilt, used by a script with no area filter)
    defaults to []."""
    active = plot_config["scripts"][script_name]["active"]
    preset = plot_config["filter_presets"][active]
    return (
        preset.get("ifilt", []), preset.get("iexcl", []),
        preset.get("afilt", []), preset.get("aexcl", []),
    )


# ---------------------------------------------------------------------------
# Beta-series helpers (shared by land-use error/pred-vs-actual plotting
# scripts, crops and Brazil alike)
# ---------------------------------------------------------------------------

def add_beta_current(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a "beta_current" column: each row's own most-recent (current_year)
    beta, looked up from its wide beta_<year> columns. Not a fixed calendar
    year, since not every row's series necessarily runs through the same
    last year. Rows whose current_year has no matching beta_<year> column
    get NaN. Returns df (mutated in place, also returned for convenience)."""
    beta_year_cols = sorted((c for c in df.columns if c.startswith("beta_")), key=lambda c: int(c.split("_")[1]))
    years_avail = [int(c.split("_")[1]) for c in beta_year_cols]
    col_of_year = {y: i for i, y in enumerate(years_avail)}
    current_year = df["current_year"].astype(int)
    col_pos = current_year.map(col_of_year)
    has_col = col_pos.notna()
    df["beta_current"] = np.nan
    row_pos = np.flatnonzero(has_col.values)
    col_pos_valid = col_pos[has_col].astype(int).values
    df.loc[has_col, "beta_current"] = df[beta_year_cols].values[row_pos, col_pos_valid]
    return df


def weighted_group_mean(df: pd.DataFrame, group_col: str, value_col: str, weight_col: str,
                         min_count: int = 1) -> pd.Series:
    """Weight-`weight_col` mean of `value_col` within each `group_col` group
    -- e.g. production-weighted mean % error per crop Item, or per Brazil
    state (uf). Groups with fewer than `min_count` rows, or non-positive
    total weight, are dropped. Returns a Series indexed by group, sorted
    descending by the weighted mean (worst/highest first)."""
    counts = df.groupby(group_col).size()

    def _weighted_mean(g):
        wt = g[weight_col].to_numpy()
        if wt.sum() <= 0:
            return np.nan
        return np.average(g[value_col], weights=wt)

    result = df.groupby(group_col).apply(_weighted_mean, include_groups=False)
    result = result[counts[result.index] >= min_count]
    return result.dropna().sort_values(ascending=False)
