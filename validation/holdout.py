"""
Held-out validation of the crop betas (_compute_beta_crops.py), as a test of
what the dataset is for: predicting the change in land use that goes with a
given change in production.

For each (Area, Item) series the final HOLDOUT_YEARS years are held out. Each
model is fit on the training years only -- the TVP model exactly as the main
pipeline fits it (fit_tvp_beta with config.json's Q prior), so its last
smoothed beta is the "current" beta the dataset would have published at the
end of training. Then, from the area at the end of training (A_T) and the
observed production in the held-out years, each model predicts the area h
years on:

    A_hat_{T+h} = A_T * exp(beta * sum_{k<=h} p_{T+k}),   p = Dln(Production)

i.e. the elasticity applied to the cumulative change in production. The
fitted drift (the training mean of a_t) is deliberately not extrapolated: a
scenario user applies beta to a production change, and on these short series
extrapolating the drift makes predictions worse.

Arms:
  tvp           TVP beta at the end of training (the published model)
  tvp_no_prior  the same with Q unregularized -- what the Q prior guards against
  static        OLS beta over the training years (_compute_beta_crops_linear.py's model)
  proportional  beta = 1: area moves in proportion to production (fixed land per tonne)
  no_change     beta = 0: area stays at A_T

The area is config.json's area_method (AREA_COL), so this validates whichever
product the compute scripts produced. Errors are reported per series as
|A_hat - A| / A (median) and across series as sum|A_hat - A| / sum A
(area-weighted: hectares, unlike tonnes, are comparable across crops). FAOSTAT
aggregate items are skipped -- they double count their component crops.

For the fitted arms a 90% interval for ln(A_{T+h}) is also checked, with p_k
the held-out production changes:

    TVP:    var = (sum_k p_k)^2 P_T + Q sum_j (sum_{k>=j} p_k)^2 + h R
            (P_T the end-of-training state variance; beta random-walks on by Q a year)
    static: var = (sum_k p_k)^2 se^2 + h sigma^2
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _stats import build_raw_diffs, fit_tvp_beta
from _utils import (
    AREA_COL, AREA_METHOD, CROP_SERIES_COLS, VALIDATION_OUTPUTS_PATH,
    load_aggregate_matcher, load_crop_wide, load_q_prior_config,
)

HOLDOUT_YEARS: int = 3
MIN_TRAIN_OBS: int = 12  # with HOLDOUT_YEARS, the same >= 16-year eligibility as _compute_beta_crops.py (MIN_OBS = 15)
Z_90: float = 1.645  # two-sided 90% normal quantile
BETA_DIFF_THRESHOLD: float = 0.05  # |beta_tvp - beta_static| above which the two are treated as materially different

ARMS = ["tvp", "tvp_no_prior", "static", "proportional", "no_change"]
INTERVAL_ARMS = ["tvp", "tvp_no_prior", "static"]

OUT_SERIES_PATH: Path = VALIDATION_OUTPUTS_PATH / "holdout_series.csv"
OUT_SUMMARY_PATH: Path = VALIDATION_OUTPUTS_PATH / "holdout_summary.csv"
OUT_TVP_VS_STATIC_PATH: Path = VALIDATION_OUTPUTS_PATH / "holdout_tvp_vs_static.csv"
OUT_BY_ITEM_PATH: Path = VALIDATION_OUTPUTS_PATH / "holdout_by_item.csv"

is_aggregate = load_aggregate_matcher()
Q_PRIOR_SCALE, Q_PRIOR_STRENGTH = load_q_prior_config()


def tvp_log_var(fit, p_ahead):
    """Predictive variance of ln(A_{T+h}) - ln(A_T) under a fit_tvp_beta fit
    on the training years, given the held-out production changes p_ahead
    (length h): end-of-training state variance, plus Q a year as beta
    random-walks on, plus R a year."""
    tail = np.cumsum(p_ahead[::-1])[::-1]  # sum_{k>=j} p_k, j = 1..h
    return p_ahead.sum() ** 2 * fit["se"][-1] ** 2 + fit["Q"] * np.sum(tail ** 2) + len(p_ahead) * fit["R"]


def split_series(g, area_col=AREA_COL):
    """g: one (Area, Item) series' rows of load_crop_wide(), sorted by Year,
    with area_col, Production and Yield present. Splits off the last
    HOLDOUT_YEARS diffs and fits the static OLS beta on the rest; returns
    None if the series doesn't qualify."""
    if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
        return None
    years = g["Year"].values.astype(int)
    with np.errstate(divide="ignore"):
        diff_years, a, p = build_raw_diffs(years, np.log(g[area_col].values), np.log(g["Production"].values))

    train_end = len(a) - HOLDOUT_YEARS
    valid = np.isfinite(a) & np.isfinite(p)
    train = valid[:train_end]
    # the held-out years must be complete: every prediction chains from A_T
    if train.sum() < MIN_TRAIN_OBS or not valid[train_end:].all():
        return None

    # demean with training means only, so nothing from the held-out years leaks into the fit
    a_tr = np.where(train, a[:train_end] - a[:train_end][train].mean(), np.nan)
    p_tr = np.where(train, p[:train_end] - p[:train_end][train].mean(), np.nan)

    x, y = p_tr[train], a_tr[train]
    sxx = x @ x
    if not sxx > 0:
        return None
    beta_static = (x @ y) / sxx
    sigma2_static = np.sum((y - beta_static * x) ** 2) / (len(x) - 2)  # slope + intercept (the demeaning)

    area = pd.Series(g[area_col].values, index=years)
    return {
        "diff_years": diff_years, "train_end": train_end, "n_train": int(train.sum()),
        "a_tr": a_tr, "p_tr": p_tr, "p_test": p[train_end:],
        "beta_static": beta_static, "sigma2_static": sigma2_static, "se2_static": sigma2_static / sxx,
        "area": area, "A_T": area[diff_years[train_end - 1]],
    }


def score_series(g, area_col=AREA_COL):
    """g: as for split_series. Returns one record per held-out horizon
    h = 1..HOLDOUT_YEARS, or [] if the series doesn't qualify."""
    s = split_series(g, area_col)
    if s is None:
        return []
    diff_years, train_end, a_tr, p_tr = s["diff_years"], s["train_end"], s["a_tr"], s["p_tr"]
    beta_static, sigma2_static, se2_static = s["beta_static"], s["sigma2_static"], s["se2_static"]

    fits = {
        "tvp": fit_tvp_beta(a_tr, p_tr, min_obs=MIN_TRAIN_OBS, q_prior_scale=Q_PRIOR_SCALE, q_prior_strength=Q_PRIOR_STRENGTH),
        "tvp_no_prior": fit_tvp_beta(a_tr, p_tr, min_obs=MIN_TRAIN_OBS),
    }
    if any(f is None for f in fits.values()):
        return []

    betas = {
        "tvp": fits["tvp"]["beta"][-1],
        "tvp_no_prior": fits["tvp_no_prior"]["beta"][-1],
        "static": beta_static,
        "proportional": 1.0,
        "no_change": 0.0,
    }
    series_info = {
        "train_start": int(diff_years[0]) - 1,
        "train_end": int(diff_years[train_end - 1]),
        "n_train": s["n_train"],
        "beta_tvp": betas["tvp"], "se_tvp": fits["tvp"]["se"][-1],
        "Q_hat": fits["tvp"]["Q"], "R_hat": fits["tvp"]["R"],
        "beta_tvp_no_prior": betas["tvp_no_prior"],
        "beta_static": beta_static, "se_static": np.sqrt(se2_static),
    }

    area, A_T, p_test = s["area"], s["A_T"], s["p_test"]

    records = []
    for h in range(1, HOLDOUT_YEARS + 1):
        year = int(diff_years[train_end - 1 + h])
        A_obs = area[year]
        p_ahead = p_test[:h]
        cum_p = p_ahead.sum()
        rec = {"h": h, "year": year, "A_T": A_T, "A_obs": A_obs, "cum_dln_production": cum_p, **series_info}
        for arm, beta in betas.items():
            A_pred = A_T * np.exp(beta * cum_p)
            rec[f"pred_{arm}"] = A_pred
            rec[f"err_{arm}"] = (A_pred - A_obs) / A_obs

        log_var = {arm: tvp_log_var(fit, p_ahead) for arm, fit in fits.items()}
        log_var["static"] = cum_p ** 2 * se2_static + h * sigma2_static
        log_ratio = np.log(A_obs / A_T)
        for arm in INTERVAL_ARMS:
            rec[f"cover90_{arm}"] = abs(log_ratio - betas[arm] * cum_p) <= Z_90 * np.sqrt(log_var[arm])
        records.append(rec)
    return records


def summarise(d):
    """One row per arm for a set of (series, horizon) rows of the results."""
    w = d["A_obs"]
    rows = []
    for arm in ARMS:
        abs_err = d[f"err_{arm}"].abs()
        row = {
            "arm": arm,
            "n_series": len(d),
            "median_abs_pct_err": abs_err.median(),
            "area_weighted_abs_err": (abs_err * w).sum() / w.sum(),  # = sum|A_hat - A| / sum A
            "median_signed_pct_err": d[f"err_{arm}"].median(),
        }
        for other in ["no_change", "proportional", "static"]:
            if other != arm:
                row[f"area_share_beating_{other}"] = (w * (abs_err < d[f"err_{other}"].abs())).sum() / w.sum()
        if arm in INTERVAL_ARMS:
            row["cover90"] = d[f"cover90_{arm}"].mean()
            row["cover90_area_weighted"] = np.average(d[f"cover90_{arm}"], weights=w)
        rows.append(row)
    return pd.DataFrame(rows)


def tvp_vs_static(d):
    """Head-to-head TVP vs static, overall and where the two betas
    materially differ (|beta_tvp - beta_static| > BETA_DIFF_THRESHOLD)."""
    rows = []
    differ = (d["beta_tvp"] - d["beta_static"]).abs() > BETA_DIFF_THRESHOLD
    for subset, m in [("all", np.ones(len(d), dtype=bool)), (f"|beta_tvp - beta_static| > {BETA_DIFF_THRESHOLD}", differ)]:
        s = d[m]
        w = s["A_obs"]
        tvp_better = s["err_tvp"].abs() < s["err_static"].abs()
        rows.append({
            "subset": subset,
            "n_series": len(s),
            "share_of_series": len(s) / len(d),
            "tvp_better_share_series": tvp_better.mean(),
            "tvp_better_share_area": (w * tvp_better).sum() / w.sum(),
            "area_weighted_abs_err_tvp": (s["err_tvp"].abs() * w).sum() / w.sum(),
            "area_weighted_abs_err_static": (s["err_static"].abs() * w).sum() / w.sum(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    wide = load_crop_wide()

    records = []
    groups = wide.groupby(["Area", "Area Code", "Item", "Item Code"])
    for i, ((area, area_code, item, item_code), g) in enumerate(groups):
        if i % 1000 == 0:
            print(i, "/", len(groups))
        if is_aggregate(item):
            continue
        g = g.sort_values("Year").dropna(subset=CROP_SERIES_COLS)
        for rec in score_series(g):
            records.append({"Area": area, "Area Code": area_code, "Item": item, "Item Code": item_code, **rec})

    res = pd.DataFrame.from_records(records)

    summary = pd.concat([summarise(d).assign(h=h) for h, d in res.groupby("h")], ignore_index=True)
    summary = summary[["h"] + [c for c in summary.columns if c != "h"]]
    head_to_head = pd.concat([tvp_vs_static(d).assign(h=h) for h, d in res.groupby("h")], ignore_index=True)
    head_to_head = head_to_head[["h"] + [c for c in head_to_head.columns if c != "h"]]

    # per item, at the full held-out horizon
    last = res[res["h"] == HOLDOUT_YEARS]
    by_item = last.groupby("Item").apply(
        lambda d: pd.Series({
            "n_series": len(d),
            "area_ha": d["A_obs"].sum(),
            **{f"area_weighted_abs_err_{arm}": (d[f"err_{arm}"].abs() * d["A_obs"]).sum() / d["A_obs"].sum() for arm in ARMS},
        }),
        include_groups=False,
    ).sort_values("area_ha", ascending=False)

    VALIDATION_OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_SERIES_PATH, index=False)
    summary.to_csv(OUT_SUMMARY_PATH, index=False)
    head_to_head.to_csv(OUT_TVP_VS_STATIC_PATH, index=False)
    by_item.to_csv(OUT_BY_ITEM_PATH)

    n_series = res.groupby(["Area Code", "Item Code"]).ngroups
    print(f"\n{AREA_METHOD}: {n_series} series ({last['A_obs'].sum() / 1e6:.0f} Mha), "
          f"last {HOLDOUT_YEARS} years held out (train >= {MIN_TRAIN_OBS} diffs).")
    print(f"\n=== Area {HOLDOUT_YEARS} years on, predicted from A_T and observed production ===")
    show = summary[summary["h"] == HOLDOUT_YEARS].set_index("arm")
    print(show[["median_abs_pct_err", "area_weighted_abs_err", "area_share_beating_no_change",
                "area_share_beating_proportional", "cover90", "cover90_area_weighted"]]
          .to_string(float_format=lambda x: f"{x:.1%}"))

    print("\n=== TVP vs static (linear) beta ===")
    print(head_to_head[head_to_head["h"] == HOLDOUT_YEARS].drop(columns="h").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    abs_diff = (last["beta_tvp"] - last["beta_static"]).abs()
    print(f"|beta_tvp - beta_static| at the end of training: median {abs_diff.median():.4f}, "
          f"<= {BETA_DIFF_THRESHOLD} in {(abs_diff <= BETA_DIFF_THRESHOLD).mean():.1%} of series")

    print(f"\nWritten to {OUT_SERIES_PATH.parent}: {OUT_SERIES_PATH.name}, {OUT_SUMMARY_PATH.name}, "
          f"{OUT_TVP_VS_STATIC_PATH.name}, {OUT_BY_ITEM_PATH.name}")
