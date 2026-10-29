"""
Sensitivity of the held-out test (validation/holdout.py) to the Q prior.
fit_tvp_beta's prior is log Q ~ N(log(q_prior_scale * var(a)), q_prior_strength^2):
q_prior_scale sets the typical yearly beta variance relative to the variance
of the area changes, q_prior_strength how far Q may stray from it (smaller =
tighter). Every (scale, strength) in the grid is fit on the same training
years of the same series, alongside no prior and the static OLS beta, and
scored at h = HOLDOUT_YEARS on the configured area_method:

  log_err        sum A |ln A_hat - ln A| / sum A  (area-weighted)
  median_log_err median over series of |ln A_hat - ln A|
  n_blowups      series with |ln A_hat - ln A| > 1 (off by more than a factor e)
  tvp_closer     share of area where this fit is closer than the static beta
  cover90        share of series inside the 90% predictive interval
  log_score      mean Gaussian predictive log density of ln(A/A_T): rewards
                 accurate and well-calibrated intervals together (higher = better)
"""

import os
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _stats import fit_tvp_beta
from _utils import AREA_METHOD, CROP_SERIES_COLS, VALIDATION_OUTPUTS_PATH, load_crop_wide, load_q_prior_config
from holdout import BETA_DIFF_THRESHOLD, MIN_TRAIN_OBS, Z_90, is_aggregate, split_series, tvp_log_var

OUT_PATH: Path = VALIDATION_OUTPUTS_PATH / "holdout_q_prior.csv"
OUT_SERIES_PATH: Path = VALIDATION_OUTPUTS_PATH / "holdout_q_prior_series.csv"

Q_PRIOR_SCALES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
Q_PRIOR_STRENGTHS = [0.3, 0.9, 2.0]
N_WORKERS: int = min(8, os.cpu_count() or 1)
KEYS = ["Area", "Area Code", "Item", "Item Code"]


def score_priors(item):
    """All prior settings for one series: one record per setting, plus the
    static beta, or [] if the series doesn't qualify."""
    key, g = item
    s = split_series(g.sort_values("Year").dropna(subset=CROP_SERIES_COLS))
    if s is None:
        return []
    p_ahead = s["p_test"]
    cum_p = p_ahead.sum()
    A_obs = s["area"][int(s["diff_years"][-1])]
    log_ratio = np.log(A_obs / s["A_T"])
    var_a = np.nanvar(s["a_tr"])

    settings = [("none", None, np.nan)] + [
        (f"{scale}/{strength}", scale, strength) for scale in Q_PRIOR_SCALES for strength in Q_PRIOR_STRENGTHS
    ]
    rows = []
    for label, scale, strength in settings:
        fit = fit_tvp_beta(s["a_tr"], s["p_tr"], min_obs=MIN_TRAIN_OBS, q_prior_scale=scale,
                           q_prior_strength=1.0 if scale is None else strength)
        if fit is None:
            return []
        rows.append({"prior": label, "q_prior_scale": scale, "q_prior_strength": strength,
                     "beta": fit["beta"][-1], "log_var": tvp_log_var(fit, p_ahead), "Q_over_var_a": fit["Q"] / var_a})
    rows.append({"prior": "static", "beta": s["beta_static"],
                 "log_var": cum_p ** 2 * s["se2_static"] + len(p_ahead) * s["sigma2_static"]})

    out = []
    for r in rows:
        resid = log_ratio - r["beta"] * cum_p
        out.append({**dict(zip(KEYS, key)), **r, "A_obs": A_obs, "beta_static": s["beta_static"],
                    "abs_log_err": abs(resid),
                    "cover90": abs(resid) <= Z_90 * np.sqrt(r["log_var"]),
                    "log_score": -0.5 * (np.log(2 * np.pi * r["log_var"]) + resid ** 2 / r["log_var"])})
    return out


def summarise_prior(d, static_err):
    w = d["A_obs"]
    closer = d["abs_log_err"].values < static_err.loc[d.set_index(KEYS).index].values
    beta_diff = (d["beta"] - d["beta_static"]).abs()
    return pd.Series({
        "log_err": (d["abs_log_err"] * w).sum() / w.sum(),
        "median_log_err": d["abs_log_err"].median(),
        "n_blowups": int((d["abs_log_err"] > 1).sum()),
        "tvp_closer_share_area": (w * closer).sum() / w.sum(),
        "median_abs_beta_diff": beta_diff.median(),
        "share_beta_differ": (beta_diff > BETA_DIFF_THRESHOLD).mean(),
        "median_Q_over_var_a": d["Q_over_var_a"].median(),
        "cover90": d["cover90"].mean(),
        "cover90_area_weighted": np.average(d["cover90"], weights=w),
        "log_score": d["log_score"].mean(),
    })


if __name__ == "__main__":
    wide = load_crop_wide()
    wide = wide[~wide["Item"].map(is_aggregate)]
    groups = list(wide.groupby(KEYS))

    records = []
    with Pool(N_WORKERS) as pool:
        for i, rows in enumerate(pool.imap_unordered(score_priors, groups, chunksize=20), 1):
            records.extend(rows)
            if i % 1000 == 0:
                print(i, "/", len(groups))
    res = pd.DataFrame.from_records(records)

    static_err = res[res["prior"] == "static"].set_index(KEYS)["abs_log_err"]
    table = res.groupby("prior", sort=False).apply(summarise_prior, static_err=static_err, include_groups=False)
    table.loc["static", ["tvp_closer_share_area", "median_Q_over_var_a"]] = np.nan

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH)
    res.to_csv(OUT_SERIES_PATH, index=False)

    scale, strength = load_q_prior_config()
    print(f"\n{AREA_METHOD}: {res.groupby(KEYS).ngroups} series; prior = q_prior_scale/q_prior_strength "
          f"(config.json: {scale}/{strength})\n")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nWritten to {OUT_PATH} and {OUT_SERIES_PATH.name}")
