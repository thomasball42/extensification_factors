"""
Per-observation actual vs. estimated values underlying
holdout_comparison_animals_brazil.py's MAE figures.

That script only ever writes the aggregate mae_static/mae_tvp per region --
useful for the headline comparison, but it throws away the individual
held-out (actual, predicted) pairs each MAE is averaged over. This script
recomputes the same static-OLS and TVP fits (identical config: MIN_TRAIN_OBS,
HOLDOUT_YEARS, USE_RESIDUALIZATION, Q prior) and additionally records, for
every region x held-out year, the actual demeaned value and both models'
one-step-ahead predictions -- the raw numbers a scatter of estimated vs.
actual (and hence the MAE distribution) needs.

Run with working directory = extensification_factors/.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

FILE_DIR = Path(__file__).resolve().parent                # .../extensification_factors/validation/brazil
VALIDATION_DIR = FILE_DIR.parent                            # .../extensification_factors/validation
EXT_FACTORS_DIR = VALIDATION_DIR.parent                      # .../extensification_factors

sys.path.insert(0, str(EXT_FACTORS_DIR))
sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(FILE_DIR))

from _stats import kalman_filter_diag, q_shrinkage_penalty
from _utils import load_q_prior_config
from _holdout_common import residualized_target_series, demeaned_target_series
from _holdout_common_brazil import load_data, load_groups, build_region_data

MIN_TRAIN_OBS = 6
HOLDOUT_YEARS = 2
SAMPLE_N = None
RANDOM_SEED = 0
USE_RESIDUALIZATION = False  # matches holdout_comparison_animals_brazil.py / _compute_beta_animals_brazil.py

Q_PRIOR_SCALE, Q_PRIOR_STRENGTH = load_q_prior_config()

OUT_PATH = Path("outputs") / "validation" / "brazil" / "holdout_estimates_vs_actual_brazil.csv"


def score_region(cod_rgi, region_entry, panel, q_prior_scale, q_prior_strength):
    """Same fit as holdout_comparison_animals_brazil.compare_on_series, but
    returns one row per held-out (region, year) observation instead of an
    aggregate MAE."""
    g = region_entry["g"]
    diff_years = region_entry["diff_years"]
    if len(g) < MIN_TRAIN_OBS + HOLDOUT_YEARS + 1:
        return None
    T = len(diff_years)
    train_end = T - HOLDOUT_YEARS
    if train_end < MIN_TRAIN_OBS:
        return None

    if USE_RESIDUALIZATION:
        a, p = residualized_target_series(cod_rgi, region_entry, panel, train_end)
    else:
        a, p = demeaned_target_series(cod_rgi, region_entry, panel, train_end)

    valid = np.isfinite(a) & np.isfinite(p)
    train_valid = valid.copy()
    train_valid[train_end:] = False
    if train_valid.sum() < MIN_TRAIN_OBS or not valid[train_end:].any():
        return None

    a_tr, p_tr = a[train_valid], p[train_valid]
    var_p_tr = np.var(p_tr)
    if not (np.isfinite(var_p_tr) and var_p_tr > 0):
        return None

    beta_static = np.sum(p_tr * a_tr) / np.sum(p_tr * p_tr)

    test_mask = np.zeros(T, dtype=bool)
    test_mask[train_end:] = True
    test_mask &= valid
    if not test_mask.any():
        return None

    var_a_tr = np.var(a_tr)
    if not (np.isfinite(var_a_tr) and var_a_tr > 0):
        return None

    def neg_ll(log_qr):
        Q, R = np.exp(log_qr)
        *_, ll = kalman_filter_diag(a, p, Q, R, fit_upto=train_end)
        if not np.isfinite(ll):
            return 1e10
        return -ll + q_shrinkage_penalty(Q, var_a_tr, q_prior_scale, q_prior_strength)

    res = minimize(neg_ll, np.log([0.05 * var_a_tr, var_a_tr]),
                    method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 500})
    if not res.success:
        return None
    Q_hat, R_hat = np.exp(res.x)
    _, v, F, _ = kalman_filter_diag(a, p, Q_hat, R_hat, fit_upto=T)

    a_test = a[test_mask]
    p_test = p[test_mask]
    v_test = v[test_mask]
    years_test = diff_years[test_mask]

    yhat_static = beta_static * p_test
    yhat_tvp = a_test - v_test  # v_t = a_t - Z_t * beta_pred(t)  ->  yhat_t = a_t - v_t

    current = g.iloc[-1]
    rows = []
    for yr, act, est_s, est_t in zip(years_test, a_test, yhat_static, yhat_tvp):
        rows.append({
            "uf": current["uf"], "cod_rgi": cod_rgi, "nome_rgi": current["nome_rgi"],
            "year": int(yr),
            "actual": float(act),
            "estimated_static": float(est_s),
            "estimated_tvp": float(est_t),
        })
    return rows


def main():
    df = load_data()
    region_ids = load_groups(df, SAMPLE_N, RANDOM_SEED)
    region_data, panel = build_region_data(df)

    results = []
    for i, cod_rgi in enumerate(region_ids):
        region_entry = region_data.get(cod_rgi)
        if region_entry is None:
            continue
        print(i + 1, "/", len(region_ids), cod_rgi)
        rows = score_region(cod_rgi, region_entry, panel, Q_PRIOR_SCALE, Q_PRIOR_STRENGTH)
        if rows is None:
            continue
        results.extend(rows)

    res_df = pd.DataFrame(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUT_PATH, index=False)

    mae_static = (res_df["actual"] - res_df["estimated_static"]).abs().mean()
    mae_tvp = (res_df["actual"] - res_df["estimated_tvp"]).abs().mean()
    print(f"\n{len(res_df)} held-out (region, year) observations from {res_df['cod_rgi'].nunique()} regions.")
    print(f"MAE -- static baseline: {mae_static:.4f}")
    print(f"MAE -- TVP:             {mae_tvp:.4f}")
    print(f"Written to {OUT_PATH}")
    return res_df


if __name__ == "__main__":
    main()
