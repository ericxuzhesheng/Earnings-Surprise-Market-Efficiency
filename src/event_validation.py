"""Pre-specified earnings diagnostics; never select a model on its p-value."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


def match_consensus(events, reports, freshness_days=365, before_announcement=True):
    """Same fiscal period, latest valid report per named broker, median in CNY 10,000."""
    groups = {key: frame for key, frame in reports.groupby(["ts_code", "period_end"])}
    rows = []
    for event in events.itertuples(index=False):
        cutoff = event.ann_date if before_announcement else event.event_trade_date
        group = groups.get((event.ts_code, event.period_end), pd.DataFrame())
        if not group.empty:
            group = group.loc[(group.report_date < cutoff)
                              & (group.report_date >= cutoff - pd.Timedelta(days=freshness_days))
                              & group.np.notna() & group.org_name.notna()].copy()
            group = group[group.org_name.astype(str).str.strip().ne("")]
            # Unknown ordering of conflicting same-day estimates is not resolved
            # using arbitrary CSV order; exclude that broker/date snapshot.
            conflicts = group.groupby(["org_name", "report_date"]).np.transform("nunique") > 1
            group = group.loc[~conflicts].sort_values("report_date").drop_duplicates("org_name", keep="last")
        expected = group.np.median() if not group.empty else np.nan
        rows.append({"event_id": event.event_id, "expected_np": expected,
                     "matched_brokers": len(group),
                     "latest_report_date": group.report_date.max() if not group.empty else pd.NaT,
                     "surprise_pct": (event.forecast_profit_mid - expected) / abs(expected)
                         if pd.notna(expected) and expected != 0 else np.nan})
    return pd.DataFrame(rows)


def calendar_returns(prices, market, factors=None):
    """Preserve market sessions and corporate actions; a missing price stays missing."""
    if prices.trade_date.duplicated().any() or market.trade_date.duplicated().any():
        raise ValueError("Duplicate price/market date")
    out = market[["trade_date", "mkt_ret"]].sort_values("trade_date").set_index("trade_date")
    price = prices.set_index("trade_date").close.reindex(out.index)
    if factors is not None:
        if factors.trade_date.duplicated().any():
            raise ValueError("Duplicate adjustment-factor date")
        factor = factors.set_index("trade_date").adj_factor.reindex(out.index)
        price = price * factor.where(factor > 0)
    out["ret"] = price.pct_change(fill_method=None)
    out["abret"] = out.ret - out.mkt_ret
    return out


def window_car(frame, event_date, start, end):
    """Day zero is the first market session strictly after date-only disclosure."""
    if event_date not in frame.index:
        return np.nan
    event_pos = frame.index.get_loc(event_date)
    lo, hi = event_pos + start, event_pos + end
    if lo < 0 or hi >= len(frame):
        return np.nan
    values = frame.abret.iloc[lo:hi + 1]
    return float(values.sum()) if len(values) == end - start + 1 and values.notna().all() else np.nan


def has_contamination(event_date, other_dates, calendar, horizon):
    """Any other earnings announcement in [-10,+horizon] contaminates the window."""
    pos = calendar.get_loc(event_date)
    lo, hi = calendar[max(0, pos - 10)], calendar[min(len(calendar) - 1, pos + horizon)]
    return any(lo <= date <= hi and date != event_date for date in other_dates)


def clustered_estimate(frame, column, difference=False):
    """Two-way firm/calendar-month cluster covariance; exploratory, not causal."""
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
    data = frame.dropna(subset=[column, "surprise_pct"]).copy()
    if difference:
        data = data.loc[data.surprise_pct.ne(0)]
    n = len(data)
    firms = pd.factorize(data.ts_code)[0]
    months = pd.factorize(pd.to_datetime(data.event_trade_date).dt.to_period("M"))[0]
    clusters = min(len(set(firms)), len(set(months)))
    result = {"n": n, "firms": len(set(firms)), "months": len(set(months)),
              "estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
              "p_value": np.nan, "inference_valid": False}
    x = np.ones((n, 1))
    if difference:
        positive = data.surprise_pct.gt(0).astype(float).to_numpy()
        if len(np.unique(positive)) < 2:
            return result
        x = np.column_stack([x, positive])
    if n == 0 or n <= x.shape[1]:
        return result
    fit = sm.OLS(data[column].to_numpy(), x).fit()
    result["estimate"] = float(fit.params[-1])
    if clusters < 10:
        return result
    variance = cov_cluster_2groups(fit, firms, months)[0][-1, -1]
    if not np.isfinite(variance) or variance <= 0:
        return result
    se = np.sqrt(variance)
    crit = t.ppf(0.975, clusters - 1)
    result.update(se=float(se), ci_low=float(fit.params[-1] - crit * se),
                  ci_high=float(fit.params[-1] + crit * se),
                  p_value=float(2 * t.sf(abs(fit.params[-1] / se), clusters - 1)), inference_valid=True)
    return result
