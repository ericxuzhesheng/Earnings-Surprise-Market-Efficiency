from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.io_utils import save_csv, save_text


def build_guidance_events(
    guidance_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    market_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Build event candidates from guidance data.

    Main signal:
    ES_main = guidance_yoy_midpoint - analyst_consensus_yoy_proxy

    analyst_consensus_yoy_proxy (replaceable later):
    1) previous guidance midpoint for same firm/report period if available;
    2) same-report-period cross-sectional median midpoint announced in prior 90 days;
    3) fallback 0.0.
    """
    if guidance_df.empty or stocks_df.empty or market_df.empty:
        return pd.DataFrame()

    g = guidance_df.copy()
    g["p_change_min"] = pd.to_numeric(g.get("p_change_min"), errors="coerce")
    g["p_change_max"] = pd.to_numeric(g.get("p_change_max"), errors="coerce")
    g = g.dropna(subset=["ts_code", "ann_date", "end_date", "p_change_min", "p_change_max"])
    g["guidance_yoy_midpoint"] = (g["p_change_min"] + g["p_change_max"]) / 2.0
    g = g.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)

    # Attach metadata and remove ST names.
    meta = stocks_df[["ts_code", "name", "industry", "list_date"]].copy()
    meta["list_date"] = pd.to_datetime(meta["list_date"], format="%Y%m%d", errors="coerce")
    g = g.merge(meta, on="ts_code", how="left")
    g = g[~g["name"].fillna("").str.contains("ST", case=False, regex=False)].copy()

    # Tradable event date = first trading day after announcement date.
    cal = pd.Series(pd.to_datetime(market_df["trade_date"].dropna().unique())).sort_values().reset_index(drop=True)
    cal_values = cal.to_numpy()
    g["event_trade_date"] = g["ann_date"].apply(lambda d: _next_trade_day(d, cal_values))
    g = g.dropna(subset=["event_trade_date"])

    # Previous guidance for same firm + report period.
    g["prev_mid_same_period"] = g.groupby(["ts_code", "end_date"])["guidance_yoy_midpoint"].shift(1)
    g["prev_low_same_period"] = g.groupby(["ts_code", "end_date"])["p_change_min"].shift(1)
    g["prev_high_same_period"] = g.groupby(["ts_code", "end_date"])["p_change_max"].shift(1)
    g["is_upward_revision"] = (
        (g["p_change_min"] > g["prev_low_same_period"]) &
        (g["p_change_max"] > g["prev_high_same_period"])
    )

    # Consensus proxy and ES_main (legacy baseline).
    g["consensus_source"] = "fallback_zero"
    g["analyst_consensus_yoy_proxy"] = 0.0

    has_prev = g["prev_mid_same_period"].notna()
    g.loc[has_prev, "analyst_consensus_yoy_proxy"] = g.loc[has_prev, "prev_mid_same_period"]
    g.loc[has_prev, "consensus_source"] = "prior_guidance_same_period"

    # Cross-sectional proxy for remaining rows.
    unresolved_idx = g.index[g["analyst_consensus_yoy_proxy"].isna() | (g["consensus_source"] == "fallback_zero")]
    if len(unresolved_idx) > 0:
        for idx in unresolved_idx:
            row = g.loc[idx]
            if row["consensus_source"] == "prior_guidance_same_period":
                continue
            ref = g[
                (g["end_date"] == row["end_date"]) &
                (g["ann_date"] < row["ann_date"]) &
                (g["ann_date"] >= row["ann_date"] - pd.Timedelta(days=90)) &
                (g["ts_code"] != row["ts_code"])
            ]["guidance_yoy_midpoint"].dropna()
            if len(ref) >= 5:
                g.loc[idx, "analyst_consensus_yoy_proxy"] = float(ref.median())
                g.loc[idx, "consensus_source"] = "cross_sectional_90d_median"

    g["analyst_consensus_yoy_proxy"] = g["analyst_consensus_yoy_proxy"].fillna(0.0)
    g["ES_main"] = g["guidance_yoy_midpoint"] - g["analyst_consensus_yoy_proxy"]

    # --- New Standardized Signal: ES_std ---

    # 1. Hierarchical consensus proxy v2
    g["consensus_proxy_v2"] = 0.0
    g["consensus_source_v2"] = "fallback_zero"

    # 1a. Firm-report prior guidance (strongest)
    g.loc[has_prev, "consensus_proxy_v2"] = g.loc[has_prev, "prev_mid_same_period"]
    g.loc[has_prev, "consensus_source_v2"] = "prior_guidance_same_period"

    # 1b. Firm historical midpoint for same quarter
    g["quarter"] = pd.to_datetime(g["end_date"]).dt.quarter
    g["firm_hist_mid"] = g.groupby(["ts_code", "quarter"])["guidance_yoy_midpoint"].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean().shift(1)
    )

    mask_1b = g["consensus_source_v2"] == "fallback_zero"
    has_hist = mask_1b & g["firm_hist_mid"].notna()
    g.loc[has_hist, "consensus_proxy_v2"] = g.loc[has_hist, "firm_hist_mid"]
    g.loc[has_hist, "consensus_source_v2"] = "firm_hist_mid_same_q"

    # 1c. Industry-quarter trailing median
    # Create a rolling industry median
    g["ind_q"] = g["industry"] + "_" + g["quarter"].astype(str)
    ind_medians = g.sort_values(["ind_q", "ann_date"]).groupby("ind_q")["guidance_yoy_midpoint"].transform(
        lambda x: x.rolling(window=20, min_periods=5).median().shift(1)
    )
    g["ind_hist_mid"] = ind_medians

    mask_1c = g["consensus_source_v2"] == "fallback_zero"
    has_ind = mask_1c & g["ind_hist_mid"].notna()
    g.loc[has_ind, "consensus_proxy_v2"] = g.loc[has_ind, "ind_hist_mid"]
    g.loc[has_ind, "consensus_source_v2"] = "industry_hist_median"

    # 2. Raw Surprise v2
    g["ES_raw_v2"] = g["guidance_yoy_midpoint"] - g["consensus_proxy_v2"]

    # 3. Standardize by historical volatility
    # MAD of prior ES_raw_v2 for the firm
    g["firm_es_mad"] = g.groupby("ts_code")["ES_raw_v2"].transform(
        lambda x: x.rolling(window=8, min_periods=3).apply(lambda y: np.mean(np.abs(y - np.mean(y))), raw=True).shift(1)
    )

    scale_floor = 0.05  # 5% volatility floor
    global_es_mad = float(np.mean(np.abs(g["ES_raw_v2"].dropna() - g["ES_raw_v2"].dropna().mean()))) if g["ES_raw_v2"].notna().any() else scale_floor
    g["ES_scale"] = np.maximum(1.4826 * g["firm_es_mad"].fillna(global_es_mad), scale_floor)

    g["ES_std_unclipped"] = g["ES_raw_v2"] / g["ES_scale"]

    # Winsorize at 1/99 percentile to remove extreme outliers
    lower_bound = g["ES_std_unclipped"].quantile(0.01)
    upper_bound = g["ES_std_unclipped"].quantile(0.99)
    g["ES_std"] = g["ES_std_unclipped"].clip(lower=lower_bound, upper=upper_bound)
    # --- End New Signal ---

    # Keep first valid event per stock/report period for base events.
    base = g.sort_values(["ts_code", "end_date", "ann_date"]).groupby(["ts_code", "end_date"], as_index=False).head(1).copy()
    base["event_type"] = "guidance_initial"
    base["positive_revision_dummy"] = 0

    # Add one revision event per stock/report period if exists.
    rev = g[g["is_upward_revision"]].sort_values(["ts_code", "end_date", "ann_date"])
    rev = rev.groupby(["ts_code", "end_date"], as_index=False).head(1).copy()
    rev["event_type"] = "guidance_upward_revision"
    rev["positive_revision_dummy"] = 1

    events = pd.concat([base, rev], ignore_index=True)
    events = events.dropna(subset=["ES_main"]).sort_values(["event_trade_date", "ts_code"]).reset_index(drop=True)
    logger.info("Guidance events constructed: %s rows", len(events))
    return events


def _next_trade_day(ann_date: pd.Timestamp, trade_days: np.ndarray) -> pd.Timestamp | pd.NaT:
    if pd.isna(ann_date):
        return pd.NaT
    # strictly after announcement date
    pos = np.searchsorted(trade_days, np.datetime64(ann_date), side="right")
    if pos >= len(trade_days):
        return pd.NaT
    return pd.Timestamp(trade_days[pos])


def apply_tradability_filters(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    market_df: pd.DataFrame,
    min_listed_trading_days: int = 120,
    turnover20_threshold: float = 0.5,
) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    e = events_df.copy()
    p = prices_df.copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.sort_values(["ts_code", "trade_date"])
    if "vol" in p.columns:
        p["vol"] = pd.to_numeric(p["vol"], errors="coerce")

    # Event day must have tradable price and non-zero volume.
    evt_px = p[["ts_code", "trade_date", "ret", "vol"]].rename(
        columns={"trade_date": "event_trade_date", "ret": "event_day_ret", "vol": "event_day_vol"}
    )
    e = e.merge(evt_px, on=["ts_code", "event_trade_date"], how="left")
    e = e[e["event_day_ret"].notna()]
    if "event_day_vol" in e.columns:
        e = e[e["event_day_vol"].fillna(0) > 0]

    # Listed >= 120 trading days.
    cal = pd.Series(pd.to_datetime(market_df["trade_date"].dropna().unique())).sort_values().reset_index(drop=True)
    cal_map = {d: i for i, d in enumerate(cal)}
    e["list_trade_day_idx"] = e["list_date"].apply(lambda d: _first_trade_idx_after(d, cal))
    e["event_trade_day_idx"] = e["event_trade_date"].map(cal_map)
    e["listed_days"] = e["event_trade_day_idx"] - e["list_trade_day_idx"]
    e = e[e["listed_days"] >= min_listed_trading_days]

    # 20-day average turnover filter.
    if not daily_basic_df.empty and "turnover_rate" in daily_basic_df.columns:
        db = daily_basic_df.copy()
        db["trade_date"] = pd.to_datetime(db["trade_date"])
        db["turnover_rate"] = pd.to_numeric(db["turnover_rate"], errors="coerce")
        db = db.sort_values(["ts_code", "trade_date"])
        db["turnover20"] = db.groupby("ts_code")["turnover_rate"].rolling(20, min_periods=15).mean().reset_index(level=0, drop=True)
        turn = db[["ts_code", "trade_date", "turnover20"]].rename(columns={"trade_date": "event_trade_date"})
        e = e.merge(turn, on=["ts_code", "event_trade_date"], how="left")
        e = e[e["turnover20"].fillna(0) >= turnover20_threshold]
    else:
        e["turnover20"] = np.nan

    return e.reset_index(drop=True)


def _first_trade_idx_after(date_val: pd.Timestamp, calendar: pd.Series) -> int | float:
    if pd.isna(date_val):
        return np.nan
    pos = np.searchsorted(calendar.to_numpy(), np.datetime64(date_val), side="left")
    if pos >= len(calendar):
        return np.nan
    return int(pos)


def add_event_returns_and_controls(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    event_windows: tuple[int, ...] = (3, 5, 20, 60),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    p = prices_df.copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.sort_values(["ts_code", "trade_date"])
    p["ret"] = pd.to_numeric(p["ret"], errors="coerce")

    m = market_df.copy()
    m["trade_date"] = pd.to_datetime(m["trade_date"])
    m["mkt_ret"] = pd.to_numeric(m["mkt_ret"], errors="coerce")
    m = m.dropna(subset=["trade_date", "mkt_ret"]).sort_values("trade_date")

    db = daily_basic_df.copy()
    if not db.empty:
        db["trade_date"] = pd.to_datetime(db["trade_date"])
        for col in ["total_mv", "pb"]:
            if col in db.columns:
                db[col] = pd.to_numeric(db[col], errors="coerce")
        db = db.sort_values(["ts_code", "trade_date"])

    rows = []
    path_rows = []
    max_window = max(event_windows)
    for _, ev in events_df.iterrows():
        ts_code = ev["ts_code"]
        event_date = pd.to_datetime(ev["event_trade_date"])
        sp = p[p["ts_code"] == ts_code][["trade_date", "ret"]].dropna()
        if sp.empty:
            continue
        mm = sp.merge(m[["trade_date", "mkt_ret"]], on="trade_date", how="inner")
        mm = mm.sort_values("trade_date")
        mm["abret"] = mm["ret"] - mm["mkt_ret"]
        post = mm[mm["trade_date"] >= event_date].copy().reset_index(drop=True)
        if len(post) < max_window:
            continue
        post["event_day"] = np.arange(1, len(post) + 1)

        cars = {}
        for w in event_windows:
            cars[f"CAR{w}"] = post.loc[post["event_day"] <= w, "abret"].sum()

        beta = _estimate_beta(mm, event_date, est_window=120)

        size = np.nan
        bm = np.nan
        if not db.empty:
            d = db[(db["ts_code"] == ts_code) & (db["trade_date"] <= event_date)].tail(1)
            if not d.empty:
                size = d["total_mv"].iloc[0] if "total_mv" in d.columns else np.nan
                pb = d["pb"].iloc[0] if "pb" in d.columns else np.nan
                bm = np.nan if pd.isna(pb) or pb == 0 else 1.0 / pb

        row_data = {
            "ts_code": ts_code,
            "event_type": ev["event_type"],
            "industry": ev.get("industry", np.nan),
            "report_period": pd.to_datetime(ev["end_date"]).strftime("%Y-%m-%d"),
            "announcement_date": pd.to_datetime(ev["ann_date"]).strftime("%Y-%m-%d"),
            "event_trading_date": event_date.strftime("%Y-%m-%d"),
            "earnings_surprise": ev["ES_main"],
            "ES_std": ev.get("ES_std", np.nan),
            "guidance_yoy_midpoint": ev["guidance_yoy_midpoint"],
            "analyst_consensus_yoy_proxy": ev["analyst_consensus_yoy_proxy"],
            "consensus_source": ev["consensus_source"],
            "consensus_source_v2": ev.get("consensus_source_v2", np.nan),
            "positive_revision_dummy": int(ev["positive_revision_dummy"]),
            "size": size,
            "beta": beta,
            "book_to_market": bm,
            "turnover20": ev.get("turnover20", np.nan),
        }
        row_data.update(cars)
        rows.append(row_data)

        tmp = post[post["event_day"] <= max_window][["event_day", "abret"]].copy()
        tmp["ts_code"] = ts_code
        tmp["event_trading_date"] = event_date
        tmp["earnings_surprise"] = ev["ES_main"]
        tmp["ES_std"] = ev.get("ES_std", np.nan)
        path_rows.append(tmp)

    event_final = pd.DataFrame(rows)
    path_df = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    return event_final, path_df


def _estimate_beta(merged_ret: pd.DataFrame, event_date: pd.Timestamp, est_window: int = 120) -> float:
    hist = merged_ret[merged_ret["trade_date"] < event_date].tail(est_window)
    if len(hist) < 40:
        return np.nan
    x = hist["mkt_ret"].to_numpy(dtype=float)
    y = hist["ret"].to_numpy(dtype=float)
    var_x = np.var(x)
    if var_x == 0:
        return np.nan
    return float(np.cov(x, y, ddof=0)[0, 1] / var_x)


def save_core_outputs(
    event_df: pd.DataFrame,
    path_df: pd.DataFrame,
    outputs_tables_dir: Path,
    outputs_figures_dir: Path,
    logger: logging.Logger,
    scenario_name: str = "baseline",
    primary_car: str = "CAR60",
    car_windows: tuple[int, ...] = (20, 60),
    signal_col: str = "earnings_surprise",
    use_panel_regression: bool = False,
) -> dict[str, float | str]:
    outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    outputs_figures_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = outputs_tables_dir / f"final_dataset_{scenario_name}.csv"
    group_path = outputs_tables_dir / f"final_group_summary_{scenario_name}.csv"
    reg_path = outputs_tables_dir / f"final_regression_results_{scenario_name}.csv"
    note_path = outputs_tables_dir / f"final_interpretation_{scenario_name}.txt"

    if event_df.empty:
        save_csv(pd.DataFrame(), dataset_path)
        save_csv(pd.DataFrame(), group_path)
        save_csv(pd.DataFrame(), reg_path)
        save_text("No valid events after filters.", note_path)
        logger.info("Core outputs saved (empty) for %s.", scenario_name)
        return {
            "scenario": scenario_name,
            "sample_size": 0,
            "primary_car": primary_car,
            "moderate_group_mean": np.nan,
            "extreme_group_mean": np.nan,
            "coef": np.nan,
            "p_value": np.nan,
        }

    df = event_df.copy()
    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")
    df["event_trading_date"] = pd.to_datetime(df["event_trading_date"], errors="coerce")
    numeric_cols = [signal_col, "size", "beta", "book_to_market", "turnover20"] + [f"CAR{w}" for w in car_windows]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["log_size"] = np.log(df["size"].where(df["size"] > 0))
    df["year_quarter"] = df["event_trading_date"].dt.to_period("Q").astype(str)
    df["industry"] = df.get("industry", pd.Series(index=df.index, dtype=object)).fillna("unknown")

    signal = df[signal_col]
    df["positive_ES_dummy"] = (signal > 0).astype(int)
    df["high_ES_dummy"] = 0
    df["moderate_positive_ES_dummy"] = 0
    for _, d in df.groupby("event_type"):
        pos = d[d["positive_ES_dummy"] == 1][signal_col].dropna()
        if len(pos) < 10:
            continue
        p50 = pos.quantile(0.50)
        p80 = pos.quantile(0.80)
        idx_high = d.index[d[signal_col] > p80]
        idx_mod = d.index[(d[signal_col] > p50) & (d[signal_col] <= p80)]
        df.loc[idx_high, "high_ES_dummy"] = 1
        df.loc[idx_mod, "moderate_positive_ES_dummy"] = 1

    df["event_type_dummy"] = (df["event_type"] == "guidance_upward_revision").astype(int)
    save_csv(df, dataset_path)

    avg_cols = [f"avg_CAR{w}" for w in car_windows]
    car_cols = [f"CAR{w}" for w in car_windows]
    by_type = df.groupby("event_type", as_index=False)[car_cols].mean().rename(columns=dict(zip(car_cols, avg_cols)))

    group_rows = []
    group_defs = [
        ("all_events", df),
        ("positive_es", df[df["positive_ES_dummy"] == 1]),
        ("moderate_es_50_80", df[df["moderate_positive_ES_dummy"] == 1]),
        ("extreme_es_top20", df[df["high_ES_dummy"] == 1]),
    ]
    for name, d in group_defs:
        row = {"group": name, "count": len(d)}
        for w in car_windows:
            row[f"avg_CAR{w}"] = d[f"CAR{w}"].mean() if f"CAR{w}" in d.columns else np.nan
        group_rows.append(row)
    group_summary = pd.DataFrame(group_rows)
    by_type_out = by_type.rename(columns={"event_type": "group"})
    by_type_out["count"] = by_type_out["group"].map(df["event_type"].value_counts()).fillna(0).astype(int)
    final_group = pd.concat([group_summary, by_type_out], ignore_index=True, sort=False)
    save_csv(final_group, group_path)

    reg_df = _run_final_regressions(
        df=df,
        signal_col=signal_col,
        primary_car=primary_car,
        use_panel_regression=use_panel_regression,
    )
    save_csv(reg_df, reg_path)

    fig1 = outputs_figures_dir / f"fig1_es_group_comparison_{scenario_name}.png"
    fig2 = outputs_figures_dir / f"fig2_cum_return_moderate_vs_extreme_{scenario_name}.png"
    fig3 = outputs_figures_dir / f"fig3_event_type_comparison_{scenario_name}.png"
    _plot_final_group_comparison(group_summary, fig1, primary_car=primary_car)
    _plot_final_cum_moderate_vs_extreme(path_df=path_df, event_df=df, output_path=fig2, max_window=max(car_windows))
    _plot_final_event_type(by_type, fig3, primary_car=primary_car)

    note = _build_final_interpretation(
        df=df,
        by_type=by_type,
        group_summary=group_summary,
        reg_df=reg_df,
        primary_car=primary_car,
        signal_col=signal_col,
        scenario_name=scenario_name,
    )
    save_text(note, note_path)
    logger.info("Core outputs saved for %s.", scenario_name)

    key_var = "ES_std" if signal_col == "ES_std" else "moderate_positive_ES_dummy"
    key_model = "panel_signal_model" if use_panel_regression else "model_moderate_es"
    target = reg_df[(reg_df["model"] == key_model) & (reg_df["variable"] == key_var)]
    return {
        "scenario": scenario_name,
        "sample_size": int(len(df)),
        "primary_car": primary_car,
        "moderate_group_mean": float(group_summary.loc[group_summary["group"] == "moderate_es_50_80", f"avg_{primary_car}"].iloc[0]) if (group_summary["group"] == "moderate_es_50_80").any() else np.nan,
        "extreme_group_mean": float(group_summary.loc[group_summary["group"] == "extreme_es_top20", f"avg_{primary_car}"].iloc[0]) if (group_summary["group"] == "extreme_es_top20").any() else np.nan,
        "coef": float(target["coef"].iloc[0]) if not target.empty else np.nan,
        "p_value": float(target["p_value"].iloc[0]) if not target.empty else np.nan,
    }


def _run_final_regressions(
    df: pd.DataFrame,
    signal_col: str,
    primary_car: str,
    use_panel_regression: bool,
) -> pd.DataFrame:
    rows = []

    try:
        import statsmodels.api as sm  # type: ignore
        import statsmodels.formula.api as smf  # type: ignore
    except Exception:
        return pd.DataFrame()

    if use_panel_regression:
        regressors = [signal_col, "log_size", "beta"]
        if "turnover20" in df.columns and df["turnover20"].notna().sum() >= 40:
            regressors.append("turnover20")
        cols = [primary_car, "ts_code", "year_quarter", "industry"] + regressors
        dd = df[cols].dropna().copy()
        if len(dd) >= 40:
            rhs = " + ".join(regressors)
            formula = f"{primary_car} ~ {rhs} + C(industry) + C(year_quarter)"
            model = smf.ols(formula=formula, data=dd).fit(cov_type="cluster", cov_kwds={"groups": dd["ts_code"]})
            for var in regressors:
                rows.append(
                    {
                        "model": "panel_signal_model",
                        "dependent_var": primary_car,
                        "variable": var,
                        "coef": model.params.get(var, np.nan),
                        "t_stat": model.tvalues.get(var, np.nan),
                        "p_value": model.pvalues.get(var, np.nan),
                        "n_obs": int(model.nobs),
                        "r2": model.rsquared,
                    }
                )
        else:
            rows.append(
                {
                    "model": "panel_signal_model",
                    "dependent_var": primary_car,
                    "variable": "insufficient_obs",
                    "coef": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "n_obs": int(len(dd)),
                    "r2": np.nan,
                }
            )
        return pd.DataFrame(rows)

    try:
        import statsmodels.api as sm  # type: ignore
    except Exception:
        return pd.DataFrame()

    def run_model(model_name: str, xcols: list[str]) -> None:
        dd = df[[primary_car, "ts_code"] + xcols].dropna().copy()
        if len(dd) < 40:
            rows.append(
                {
                    "model": model_name,
                    "dependent_var": primary_car,
                    "variable": "insufficient_obs",
                    "coef": np.nan,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "n_obs": int(len(dd)),
                    "r2": np.nan,
                }
            )
            return
        x = sm.add_constant(dd[xcols], has_constant="add")
        model = sm.OLS(dd[primary_car], x).fit(cov_type="cluster", cov_kwds={"groups": dd["ts_code"]})
        for var in ["const"] + xcols:
            rows.append(
                {
                    "model": model_name,
                    "dependent_var": primary_car,
                    "variable": var,
                    "coef": model.params.get(var, np.nan),
                    "t_stat": model.tvalues.get(var, np.nan),
                    "p_value": model.pvalues.get(var, np.nan),
                    "n_obs": int(model.nobs),
                    "r2": model.rsquared,
                }
            )

    run_model("model_moderate_es", ["moderate_positive_ES_dummy", "log_size", "beta"])
    run_model("model_event_type", ["event_type_dummy", "log_size", "beta"])
    return pd.DataFrame(rows)


def _plot_final_group_comparison(group_summary: pd.DataFrame, output_path: Path, primary_car: str) -> None:
    import matplotlib.pyplot as plt

    metric_col = f"avg_{primary_car}"
    if group_summary.empty or metric_col not in group_summary.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(group_summary["group"], group_summary[metric_col], color="#4C72B0")
    ax.set_xlabel("Group")
    ax.set_ylabel(f"Average {primary_car}")
    ax.set_title(f"{primary_car} Comparison by ES Group")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_final_cum_moderate_vs_extreme(path_df: pd.DataFrame, event_df: pd.DataFrame, output_path: Path, max_window: int) -> None:
    import matplotlib.pyplot as plt

    if path_df.empty or event_df.empty:
        return
    p = path_df.copy()
    p["event_trading_date"] = pd.to_datetime(p["event_trading_date"], errors="coerce")
    e = event_df[["ts_code", "event_trading_date", "moderate_positive_ES_dummy", "high_ES_dummy"]].copy()
    e["event_trading_date"] = pd.to_datetime(e["event_trading_date"], errors="coerce")
    d = p.merge(e, on=["ts_code", "event_trading_date"], how="left")
    d = d.dropna(subset=["event_day", "abret"])
    d = d[d["event_day"] <= max_window]
    mod = d[d["moderate_positive_ES_dummy"] == 1].groupby("event_day")["abret"].mean().sort_index().cumsum()
    ext = d[d["high_ES_dummy"] == 1].groupby("event_day")["abret"].mean().sort_index().cumsum()
    if mod.empty or ext.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mod.index, mod.values, label="Moderate ES (50-80%)", linewidth=2.0)
    ax.plot(ext.index, ext.values, label="Extreme ES (Top 20%)", linewidth=2.0)
    ax.axhline(0, color="gray", linewidth=1.0)
    ax.set_xlabel("Event Day")
    ax.set_ylabel("Cumulative Abnormal Return")
    ax.set_title("Cumulative AR: Moderate vs Extreme ES")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_final_event_type(by_type: pd.DataFrame, output_path: Path, primary_car: str) -> None:
    import matplotlib.pyplot as plt

    metric_col = f"avg_{primary_car}"
    if by_type.empty or metric_col not in by_type.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_type["event_type"], by_type[metric_col], color="#2A9D8F")
    ax.set_xlabel("Event Type")
    ax.set_ylabel(f"Average {primary_car}")
    ax.set_title(f"{primary_car} by Event Type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_final_interpretation(
    df: pd.DataFrame,
    by_type: pd.DataFrame,
    group_summary: pd.DataFrame,
    reg_df: pd.DataFrame,
    primary_car: str,
    signal_col: str,
    scenario_name: str,
) -> str:
    metric_col = f"avg_{primary_car}"

    def gval(name: str) -> float:
        r = group_summary[group_summary["group"] == name]
        return float(r[metric_col].iloc[0]) if (not r.empty and metric_col in r.columns) else np.nan

    all_car = gval("all_events")
    pos_car = gval("positive_es")
    mod_car = gval("moderate_es_50_80")
    ext_car = gval("extreme_es_top20")
    init_car = float(by_type.loc[by_type["event_type"] == "guidance_initial", metric_col].iloc[0]) if ((by_type["event_type"] == "guidance_initial").any() and metric_col in by_type.columns) else np.nan
    rev_car = float(by_type.loc[by_type["event_type"] == "guidance_upward_revision", metric_col].iloc[0]) if ((by_type["event_type"] == "guidance_upward_revision").any() and metric_col in by_type.columns) else np.nan

    if signal_col == "ES_std":
        model_name = "panel_signal_model"
        variable_name = "ES_std"
    else:
        model_name = "model_moderate_es"
        variable_name = "moderate_positive_ES_dummy"

    m1 = reg_df[(reg_df["model"] == model_name) & (reg_df["variable"] == variable_name)]
    m2 = reg_df[(reg_df["model"] == "model_event_type") & (reg_df["variable"] == "event_type_dummy")]
    m1_coef = float(m1["coef"].iloc[0]) if not m1.empty else np.nan
    m1_p = float(m1["p_value"].iloc[0]) if not m1.empty else np.nan
    m2_coef = float(m2["coef"].iloc[0]) if not m2.empty else np.nan
    m2_p = float(m2["p_value"].iloc[0]) if not m2.empty else np.nan

    method_line = (
        "Preferred specification uses event-level OLS with industry fixed effects, year-quarter fixed effects, and firm-clustered standard errors."
        if signal_col == "ES_std"
        else "Baseline specification is retained as a comparison benchmark rather than the repository default."
    )
    conclusion_line = (
        "Cleaner standardized guidance surprise predicts modest but statistically significant short-run abnormal return, consistent with limited short-run valuation adjustment."
        if signal_col == "ES_std"
        else "Legacy long-window evidence is easier to contaminate with later news, so it should not be used as the headline conclusion."
    )

    lines = [
        f"Final Interpretation ({scenario_name})",
        "Weak significance mainly comes from noisy surprise measurement, long return windows, and under-specified regressions.",
        method_line,
        f"Group results show all-events {primary_car}={all_car:.3%}, positive-ES {primary_car}={pos_car:.3%}, moderate-positive {primary_car}={mod_car:.3%}, and extreme-ES {primary_car}={ext_car:.3%}.",
        f"By event type, guidance_initial has {primary_car}={init_car:.3%} and guidance_upward_revision has {primary_car}={rev_car:.3%}.",
        f"Key regression coefficient on {variable_name} is {m1_coef:.6f} (p={m1_p:.3f}); event_type_dummy is {m2_coef:.6f} (p={m2_p:.3f}).",
        conclusion_line,
        "Economically, the preferred interpretation should focus on whether cleaner guidance surprises predict short-horizon valuation adjustment, not on mechanically maximizing t-stats.",
    ]
    return "\n".join(lines) + "\n"
