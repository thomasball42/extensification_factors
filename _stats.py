import numpy as np
import pandas as pd
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


def q_shrinkage_penalty(Q, var_a, q_prior_scale, q_prior_strength=1.0):
    """MAP penalty pulling log(Q) toward log(q_prior_scale * var_a).

    Returns 0.0 (no-op) when q_prior_scale is None -- this is the sole
    opt-in gate, so callers can pass it unconditionally without their own
    branch and the unregularized MLE is reproduced exactly by default.

    CAUTION: q_prior_strength is the standard deviation of log(Q) under
    the prior, not a "strength" multiplier -- SMALLER q_prior_strength
    means a tighter prior (more shrinkage), LARGER means weaker.

    var_a must be the variance of the exact data window the enclosing
    log-likelihood is computed over (e.g. the full series in
    fit_tvp_beta, or the training-only slice in a train/test holdout
    fit) -- that's what keeps the prior calibrated to the current fit's
    scale.
    """
    if q_prior_scale is None:
        return 0.0
    prior_mean_log_q = np.log(q_prior_scale * var_a)
    return (np.log(Q) - prior_mean_log_q) ** 2 / (2 * q_prior_strength ** 2)


def fit_tvp_beta(a, p, P0=1e6, min_obs=15, q_prior_scale=None, q_prior_strength=1.0):
    """
    Fit Q, R by MLE, then return smoothed beta_t and its SE for every t.
    a, p: 1-D arrays of equal length, calendar-year-aligned, NaN for gaps.
    Returns None if there aren't enough valid points to identify the model.

    q_prior_scale, q_prior_strength: optional shrinkage prior on Q (see
    q_shrinkage_penalty). q_prior_scale=None (default) reproduces the
    exact unregularized MLE used previously -- fully backward compatible.
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
        return -ll + q_shrinkage_penalty(Q, var_a, q_prior_scale, q_prior_strength)

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
# Two-way (country + year) fixed-effects residualization
# ---------------------------------------------------------------------------

def residualize_two_way_fe(panel, entity_col="entity", time_col="time", value_cols=("a", "p")):
    """
    Residualize each of value_cols against two-way (entity, time) fixed effects,
    fit jointly across the whole panel (e.g. every country growing one commodity)
    via linearmodels.PanelOLS. This absorbs shocks common to a year (e.g. a global
    price spike) and each entity's own average level, leaving only the "within"
    variation -- see build_annual_diffs, which this replaces as the main pipeline's
    demeaning step.

    Rows that are the sole observation for their entity or their sole observation
    for their time period are dropped before fitting (iteratively, since removing
    one such row can turn another row into a new singleton) -- otherwise that row's
    own fixed effect would absorb it exactly, silently forcing its residual to ~0
    rather than leaving it as the unidentified value it actually is.

    Returns the input frame with "<col>_resid" columns added (NaN for rows with any
    missing value_col, dropped singletons, or if there isn't enough entity/time
    variation left to identify the fixed effects at all).
    """
    from linearmodels.panel import PanelOLS

    panel = panel.reset_index(drop=True)
    valid = np.isfinite(panel[list(value_cols)].to_numpy(dtype=float)).all(axis=1)
    sub = panel.loc[valid, [entity_col, time_col, *value_cols]].copy()

    keep = pd.Series(True, index=sub.index)
    while True:
        active = sub.loc[keep]
        big_enough = (
            active.groupby(entity_col)[entity_col].transform("size").ge(2)
            & active.groupby(time_col)[time_col].transform("size").ge(2)
        )
        if big_enough.all():
            break
        keep.loc[active.index[~big_enough.values]] = False

    fit_rows = sub.loc[keep]
    resid_cols = [f"{c}_resid" for c in value_cols]

    if fit_rows[entity_col].nunique() < 2 or fit_rows[time_col].nunique() < 2:
        for c in resid_cols:
            panel[c] = np.nan
        return panel

    indexed = fit_rows.set_index([entity_col, time_col])
    exog = pd.DataFrame({"const": 1.0}, index=indexed.index)

    resids = {}
    for col in value_cols:
        res = PanelOLS(indexed[col], exog, entity_effects=True, time_effects=True).fit()
        resids[f"{col}_resid"] = res.resids

    resid_df = pd.concat(resids, axis=1).reset_index()
    return panel.merge(resid_df, on=[entity_col, time_col], how="left")


def _fit_two_way_effects(values, entities, times, tol=1e-9, max_iter=1000):
    """Alternating-projections (iterative demeaning) fit of
    values ~ mu + alpha[entity] + gamma[time]. Mathematically equivalent to
    two-way fixed-effects OLS with only a constant regressor (Guimaraes &
    Portugal, 2010) -- used instead of linearmodels.PanelOLS here because
    residualize_two_way_fe_oos needs effects fit on one subset of rows to
    predict/residualize a DIFFERENT (held-out) subset, which requires the
    effects themselves rather than just in-sample residuals. Converges for
    any panel where every entity/time is connected to the rest via shared
    observations (the same connectivity PanelOLS's two-way FE requires).
    """
    values = np.asarray(values, dtype=float)
    ent_codes, ent_labels = pd.factorize(entities)
    time_codes, time_labels = pd.factorize(times)
    n_e, n_t = len(ent_labels), len(time_labels)

    mu = float(values.mean())
    y = values - mu
    ent_count = np.bincount(ent_codes, minlength=n_e)
    time_count = np.bincount(time_codes, minlength=n_t)

    alpha = np.zeros(n_e)
    gamma = np.zeros(n_t)
    for _ in range(max_iter):
        new_alpha = np.bincount(ent_codes, weights=y - gamma[time_codes], minlength=n_e) / ent_count
        new_gamma = np.bincount(time_codes, weights=y - new_alpha[ent_codes], minlength=n_t) / time_count
        shift = new_gamma.mean()
        new_gamma = new_gamma - shift
        new_alpha = new_alpha + shift
        delta = max(np.abs(new_alpha - alpha).max(), np.abs(new_gamma - gamma).max())
        alpha, gamma = new_alpha, new_gamma
        if delta < tol:
            break

    return mu, pd.Series(alpha, index=ent_labels), pd.Series(gamma, index=time_labels)


def residualize_two_way_fe_oos(panel, fit_mask, entity_col="entity", time_col="time", value_cols=("a", "p")):
    """
    Like residualize_two_way_fe, but fits the two-way (entity, time) fixed
    effects using only rows where fit_mask is True, then applies the fitted
    effects to compute residuals for EVERY row, including fit_mask==False
    ones. This is what a leak-safe holdout test needs: exclude a target
    entity's own held-out (test) rows from the fixed-effects estimation --
    so its own held-out values can't leak into its own entity effect -- while
    still using other entities' contemporaneous (same-year) rows to identify
    the year effects needed to residualize those held-out rows, mirroring
    what would genuinely be observable in real time.

    Same singleton-dropping rule as residualize_two_way_fe (applied only to
    the fit_mask==True rows): an entity/time with fewer than 2 fit-eligible
    rows can't have its effect identified there, so it's dropped from the
    fit; any row whose entity or time effect ends up unidentified gets a NaN
    residual.

    Returns the input frame with "<col>_resid" columns added.
    """
    panel = panel.reset_index(drop=True)
    fit_mask = np.asarray(fit_mask, dtype=bool)

    valid = np.isfinite(panel[list(value_cols)].to_numpy(dtype=float)).all(axis=1)
    keep = pd.Series(valid & fit_mask, index=panel.index)

    while keep.any():
        active_idx = panel.index[keep]
        active = panel.loc[active_idx]
        big_enough = (
            active.groupby(entity_col)[entity_col].transform("size").ge(2)
            & active.groupby(time_col)[time_col].transform("size").ge(2)
        )
        if big_enough.all():
            break
        keep.loc[active_idx[~big_enough.to_numpy()]] = False

    fit_idx = panel.index[keep]
    resid_cols = [f"{c}_resid" for c in value_cols]

    if len(fit_idx) == 0 or panel.loc[fit_idx, entity_col].nunique() < 2 or panel.loc[fit_idx, time_col].nunique() < 2:
        for c in resid_cols:
            panel[c] = np.nan
        return panel

    entity_vals = panel[entity_col].to_numpy()
    time_vals = panel[time_col].to_numpy()

    for col, rc in zip(value_cols, resid_cols):
        mu, alpha, gamma = _fit_two_way_effects(
            panel.loc[fit_idx, col].to_numpy(dtype=float),
            panel.loc[fit_idx, entity_col].to_numpy(),
            panel.loc[fit_idx, time_col].to_numpy(),
        )
        pred = mu + alpha.reindex(entity_vals).to_numpy() + gamma.reindex(time_vals).to_numpy()
        resid = panel[col].to_numpy(dtype=float) - pred
        resid[~valid] = np.nan
        panel[rc] = resid

    return panel
