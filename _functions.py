import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Kalman filter / RTS smoother for the scalar local-level regression model
# ---------------------------------------------------------------------------

def kalman_filter(a, p, Q, R, P0=1e6, beta0=0.5):
    """
    Forward filter for:
        a_t = p_t * beta_t + eps_t,  eps_t ~ N(0, R)
        beta_t = beta_{t-1} + eta_t, eta_t ~ N(0, Q)

    a[t] or p[t] == NaN is treated as a missing observation: the state
    prediction is carried forward (uncertainty grows by Q) but no update
    happens and no log-likelihood contribution is made for that step.

    Returns filtered beta/P, the one-step-ahead predicted beta/P (needed by
    the smoother), and the total log-likelihood over non-missing steps.
    """
    T = len(a)
    beta_filt = np.empty(T)
    P_filt = np.empty(T)
    beta_pred_arr = np.empty(T)
    P_pred_arr = np.empty(T)

    beta_prev, P_prev = beta0, P0
    loglik = 0.0

    for t in range(T):
        beta_pred = beta_prev
        P_pred = P_prev + Q
        beta_pred_arr[t] = beta_pred
        P_pred_arr[t] = P_pred

        obs_ok = np.isfinite(a[t]) and np.isfinite(p[t])
        if obs_ok:
            Z = p[t]
            F = Z * Z * P_pred + R
            v = a[t] - Z * beta_pred
            K = P_pred * Z / F
            beta_t = beta_pred + K * v
            P_t = P_pred - K * Z * P_pred
            loglik += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
        else:
            beta_t = beta_pred
            P_t = P_pred

        beta_filt[t] = beta_t
        P_filt[t] = P_t
        beta_prev, P_prev = beta_t, P_t

    return beta_filt, P_filt, beta_pred_arr, P_pred_arr, loglik


def rts_smoother(beta_filt, P_filt, Q):
    """Backward RTS smoother for the random-walk (transition = 1) case."""
    T = len(beta_filt)
    beta_smooth = np.empty(T)
    P_smooth = np.empty(T)

    beta_smooth[-1] = beta_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):
        P_pred_next = P_filt[t] + Q
        J = P_filt[t] / P_pred_next if P_pred_next > 0 else 0.0
        beta_smooth[t] = beta_filt[t] + J * (beta_smooth[t + 1] - beta_filt[t])
        P_smooth[t] = P_filt[t] + J * J * (P_smooth[t + 1] - P_pred_next)

    return beta_smooth, P_smooth


def fit_tvp_beta(a, p, P0=1e6, min_obs=15):
    """
    Fit Q, R by MLE, then return smoothed beta_t and its SE for every t.
    a, p: 1-D arrays of equal length, calendar-year-aligned, NaN for gaps.
    Returns None if there aren't enough valid points to identify the model.
    """
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)

    valid = np.isfinite(a) & np.isfinite(p)
    if valid.sum() < min_obs:
        return None

    var_a = np.nanvar(a[valid])
    if not (np.isfinite(var_a) and var_a > 0):
        return None

    def neg_loglik(log_params):
        Q, R = np.exp(log_params)
        _, _, _, _, ll = kalman_filter(a, p, Q, R, P0=P0)
        if not np.isfinite(ll):
            return 1e10
        return -ll

    x0 = np.log([0.05 * var_a, var_a])
    res = minimize(
        neg_loglik, x0, method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 500},
    )
    if not res.success:
        return None

    Q_hat, R_hat = np.exp(res.x)
    beta_filt, P_filt, _, _, _ = kalman_filter(a, p, Q_hat, R_hat, P0=P0)
    beta_smooth, P_smooth = rts_smoother(beta_filt, P_filt, Q_hat)

    return {
        "beta": beta_smooth,
        "se": np.sqrt(np.clip(P_smooth, 0, None)),
        "Q": Q_hat,
        "R": R_hat,
    }


def kalman_filter_diag(a, p, Q, R, P0=1e6, beta0=0.0, fit_upto=None):
    """Same recursion as kalman_filter, extended for validation use: returns
    per-step (v_t, F_t) one-step-ahead innovations/variances, and restricts
    the log-likelihood sum to indices < fit_upto (e.g. a training window)
    while still filtering through the full series -- used to fit on train
    data only and then score predictions on held-out test data."""
    T = len(a)
    if fit_upto is None:
        fit_upto = T
    beta_filt = np.empty(T)
    v_arr = np.full(T, np.nan)
    F_arr = np.full(T, np.nan)
    beta_prev, P_prev = beta0, P0
    loglik = 0.0
    for t in range(T):
        beta_pred = beta_prev
        P_pred = P_prev + Q
        if np.isfinite(a[t]) and np.isfinite(p[t]):
            Z = p[t]
            F = Z * Z * P_pred + R
            v = a[t] - Z * beta_pred
            K = P_pred * Z / F
            beta_t = beta_pred + K * v
            P_t = P_pred - K * Z * P_pred
            v_arr[t], F_arr[t] = v, F
            if t < fit_upto:
                loglik += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
        else:
            beta_t, P_t = beta_pred, P_pred
        beta_filt[t] = beta_t
        beta_prev, P_prev = beta_t, P_t
    return beta_filt, v_arr, F_arr, loglik


def build_raw_diffs(years, log_AH, log_P):
    """Reindex onto full calendar years and difference -- NOT demeaned here.
    For train/test validation, demeaning must happen after the train/test
    split (using train-only statistics) to avoid leaking test-period
    information into the fit; see build_annual_diffs for the non-split case
    used by the main pipeline, which demeans over all available points."""
    full_years = np.arange(int(years.min()), int(years.max()) + 1)
    ah = pd.Series(log_AH, index=years).reindex(full_years)
    p_ = pd.Series(log_P, index=years).reindex(full_years)
    a_t = ah.diff().values[1:].copy()
    p_t = p_.diff().values[1:].copy()
    return full_years[1:], a_t, p_t


def build_annual_diffs(years, log_AH, log_P):
    """
    Reindex onto the full calendar-year range (filling gaps with NaN) before
    differencing, so a genuine multi-year gap grows the state uncertainty
    across that gap instead of being silently compressed into a 1-year step.
    """
    full_years = np.arange(int(years.min()), int(years.max()) + 1)

    ah = pd.Series(log_AH, index=years).reindex(full_years)
    p_ = pd.Series(log_P, index=years).reindex(full_years)

    a_t = np.array(ah.diff().values[1:], dtype=float)
    p_t = np.array(p_.diff().values[1:], dtype=float)
    diff_years = full_years[1:]

    # demean over available points -- equivalent to fitting a fixed intercept
    # in a constant-slope OLS regression; see note in module docstring
    valid = np.isfinite(a_t) & np.isfinite(p_t)
    if valid.sum() > 0:
        a_t = a_t - np.nanmean(a_t[valid])
        p_t = p_t - np.nanmean(p_t[valid])
    a_t[~valid] = np.nan
    p_t[~valid] = np.nan

    return diff_years, a_t, p_t


# ---------------------------------------------------------------------------
# FAOSTAT aggregate-item flagging
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"


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
