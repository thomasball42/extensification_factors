"""
The held-out test of validation/holdout.py run three ways on the same
(country, crop) series and the same held-out years, to show how the
physical-area result compares with the harvested-area one:

  physical, 2001-       Physical area on the years with a cropping intensity
                        (what holdout.py tests under area_method = "physical_area")
  harvested, 2001-      Area harvested on exactly those years: differs only in
                        the area definition
  harvested, full       Area harvested from the series' first FAOSTAT year
                        (1961 at the earliest) to the same end year: the
                        training length the harvested-area product had

The held-out years are each series' last HOLDOUT_YEARS years with a
physical area (2017-2019 for most), and only series that qualify in all three
versions with the same held-out years are kept. One row per version, at
h = HOLDOUT_YEARS, with errors area-weighted in that version's own hectares:

  log_err_<arm>  sum A |ln A_hat - ln A| / sum A  (~ % error for small errors)
  err_<arm>      sum |A_hat - A| / sum A          (holdout.py's measure)

err_ is unbounded above, so one series whose beta blows up can carry most of
an arm's total (e.g. "harvested, full" TVP: New Zealand peaches, production
-93% with beta -3.2 predicts 49 Mha against 660 ha observed). log_err_ caps
that series' weight at its own area times its log error.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _utils import load_crop_wide
from holdout import ARMS, BETA_DIFF_THRESHOLD, HOLDOUT_YEARS, is_aggregate, score_series

OUT_PATH: Path = Path("outputs") / "physical_area" / "validation" / "holdout_area_methods.csv"
OUT_SERIES_PATH: Path = OUT_PATH.with_name("holdout_area_methods_series.csv")

VERSIONS = {  # name: (area column, training)
    "physical, 2001-": ("Physical area", "window"),
    "harvested, 2001-": ("Area harvested", "window"),
    "harvested, full": ("Area harvested", "full"),
}
KEYS = ["Area", "Area Code", "Item", "Item Code"]


def weighted_err(d, arm, log=True):
    e = np.log(d[f"pred_{arm}"] / d["A_obs"]) if log else d[f"err_{arm}"]
    return (e.abs() * d["A_obs"]).sum() / d["A_obs"].sum()


def summarise_version(d):
    w = d["A_obs"]
    differ = (d["beta_tvp"] - d["beta_static"]).abs() > BETA_DIFF_THRESHOLD
    tvp_better = d["err_tvp"].abs() < d["err_static"].abs()
    dd = d[differ]
    return pd.Series({
        "n_series": len(d),
        "first_train_year_median": d["train_start"].median(),
        "n_train_median": d["n_train"].median(),
        "area_mha": w.sum() / 1e6,
        **{f"log_err_{arm}": weighted_err(d, arm) for arm in ARMS},
        **{f"err_{arm}": weighted_err(d, arm, log=False) for arm in ARMS},
        "median_err_tvp": d["err_tvp"].abs().median(),
        "median_err_static": d["err_static"].abs().median(),
        "tvp_better_share_area": (w * tvp_better).sum() / w.sum(),
        "median_abs_beta_diff": (d["beta_tvp"] - d["beta_static"]).abs().median(),
        "differ_share_series": differ.mean(),
        "differ_tvp_better_share_area": (dd["A_obs"] * tvp_better[differ]).sum() / dd["A_obs"].sum(),
        "differ_log_err_tvp": weighted_err(dd, "tvp"),
        "differ_log_err_static": weighted_err(dd, "static"),
        "cover90_tvp": d["cover90_tvp"].mean(),
        "cover90_static": d["cover90_static"].mean(),
    })


if __name__ == "__main__":
    wide = load_crop_wide(with_physical=True)
    wide = wide[~wide["Item"].map(is_aggregate)]

    records = []
    groups = wide.groupby(KEYS)
    for i, (key, g) in enumerate(groups):
        if i % 1000 == 0:
            print(i, "/", len(groups))
        g = g.sort_values("Year")
        window = g.dropna(subset=["Physical area", "Production", "Yield"])
        if window.empty:
            continue
        full = g[g["Year"] <= window["Year"].max()].dropna(subset=["Area harvested", "Production", "Yield"])
        rows = {name: score_series(window if training == "window" else full, area_col)
                for name, (area_col, training) in VERSIONS.items()}
        held_out = {name: [r["year"] for r in recs] for name, recs in rows.items()}
        if any(not recs for recs in rows.values()) or len({tuple(y) for y in held_out.values()}) > 1:
            continue
        for name, recs in rows.items():
            records.extend({"version": name, **dict(zip(KEYS, key)), **r} for r in recs)

    res = pd.DataFrame.from_records(records)
    last = res[res["h"] == HOLDOUT_YEARS]
    table = last.groupby("version", sort=False).apply(summarise_version, include_groups=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH)
    res.to_csv(OUT_SERIES_PATH, index=False)
    print(f"\nSame series and held-out years in every version; h = {HOLDOUT_YEARS}, area-weighted errors\n")
    print(table.T.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to {OUT_PATH} and {OUT_SERIES_PATH.name}")
