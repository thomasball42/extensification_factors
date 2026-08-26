import json
from pathlib import Path

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
