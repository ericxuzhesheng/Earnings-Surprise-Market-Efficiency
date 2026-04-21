from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.expectation_alignment import match_expectations_to_events, build_sell_side_revision_panel
from src.guidance_design import _estimate_beta, _first_trade_idx_after, _next_trade_day, build_guidance_events


def build_tushare_events(
    stocks_df: pd.DataFrame,
    market_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    express_df: pd.DataFrame,
    fina_df: pd.DataFrame,
    report_rc_df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stocks_df.empty or market_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cal = pd.Series(pd.to_datetime(market_df["trade_date"].dropna().unique())).sort_values().reset_index(drop=True)
    cal_values = cal.to_numpy()
    meta = stocks_df[["ts_code", "name", "industry", "list_date"]].copy()
    meta["list_date"] = pd.to_datetime(meta["list_date"], format="%Y%m%d", errors="coerce")

    event_frames: list[pd.DataFrame] = []

    if not forecast_df.empty:
        f = forecast_df.copy()
        f = f.merge(meta, on="ts_code", how="left")
        f = f[~f["name"].fillna("").str.contains("ST", case=False, regex=False)].copy()
        f["event_trade_date"] = f["ann_date"].apply(lambda d: _next_trade_day(d, cal_values))
        f = f.dropna(subset=["event_trade_date"]).copy()
        f["period_end"] = pd.to_datetime(f["end_date"])
        f["event_type"] = np.where(f["is_revision"], "revision", "preannouncement")
        f["event_subtype"] = np.where(f["is_revision"], f["revision_direction"], "initial")
        f["event_value_np"] = f["forecast_profit_mid"]
        f["event_value_eps"] = np.nan
        f["event_value_yoy"] = f["guidance_yoy_midpoint"]
        f["event_source"] = f.get("source_endpoint", "forecast")
        event_frames.append(
            f[
                [
                    "ts_code",
                    "ann_date",
                    "end_date",
                    "period_end",
                    "event_trade_date",
                    "event_type",
                    "event_subtype",
                    "event_value_np",
                    "event_value_eps",
                    "event_value_yoy",
                    "event_source",
                    "type",
                    "p_change_min",
                    "p_change_max",
                    "net_profit_min",
                    "net_profit_max",
                    "first_ann_date",
                    "summary",
                    "change_reason",
                    "guidance_yoy_midpoint",
                    "forecast_profit_mid",
                    "revision_magnitude",
                    "source_endpoint",
                    "name",
                    "industry",
                    "list_date",
                ]
            ].copy()
        )

    if not express_df.empty:
        e = express_df.copy()
        e = e.merge(meta, on="ts_code", how="left")
        e = e[~e["name"].fillna("").str.contains("ST", case=False, regex=False)].copy()
        e["event_trade_date"] = e["ann_date"].apply(lambda d: _next_trade_day(d, cal_values))
        e = e.dropna(subset=["event_trade_date"]).copy()
        e["period_end"] = pd.to_datetime(e["end_date"])
        e["event_type"] = "express"
        e["event_subtype"] = "release"
        e["event_value_np"] = e.get("n_income", np.nan)
        e["event_value_eps"] = e.get("diluted_eps", np.nan)
        e["event_value_yoy"] = e.get("yoy_net_profit", np.nan)
        e["event_source"] = e.get("source_endpoint", "express")
        event_frames.append(
            e[
                [
                    "ts_code",
                    "ann_date",
                    "end_date",
                    "period_end",
                    "event_trade_date",
                    "event_type",
                    "event_subtype",
                    "event_value_np",
                    "event_value_eps",
                    "event_value_yoy",
                    "event_source",
                    "perf_summary",
                    "update_flag",
                    "source_endpoint",
                    "name",
                    "industry",
                    "list_date",
                ]
            ].copy()
        )

    if not fina_df.empty:
        ff = fina_df.copy()
        ff = ff.merge(meta, on="ts_code", how="left")
        ff = ff[~ff["name"].fillna("").str.contains("ST", case=False, regex=False)].copy()
        ff["event_trade_date"] = ff["ann_date"].apply(lambda d: _next_trade_day(d, cal_values))
        ff = ff.dropna(subset=["event_trade_date"]).copy()
        ff["period_end"] = pd.to_datetime(ff["end_date"])
        ff["event_type"] = "formal_release"
        ff["event_subtype"] = "release"
        ff["event_value_np"] = ff.get("actual_np", np.nan)
        ff["event_value_eps"] = ff.get("actual_eps", np.nan)
        ff["event_value_yoy"] = ff.get("dt_netprofit_yoy", ff.get("netprofit_yoy", np.nan))
        ff["event_source"] = ff.get("source_endpoint", "fina_indicator")
        event_frames.append(
            ff[
                [
                    "ts_code",
                    "ann_date",
                    "end_date",
                    "period_end",
                    "event_trade_date",
                    "event_type",
                    "event_subtype",
                    "event_value_np",
                    "event_value_eps",
                    "event_value_yoy",
                    "event_source",
                    "source_endpoint",
                    "name",
                    "industry",
                    "list_date",
                ]
            ].copy()
        )

    if not event_frames:
        return pd.DataFrame(), pd.DataFrame()

    events = pd.concat(event_frames, ignore_index=True, sort=False)
    events = events.sort_values(["event_trade_date", "ts_code", "period_end", "ann_date"]).reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)

    matched_events, expectation_audit = match_expectations_to_events(events, report_rc_df, config)
    revision_panel = build_sell_side_revision_panel(report_rc_df)
    if not revision_panel.empty:
        matched_events = matched_events.merge(
            revision_panel,
            left_on=["ts_code", "period_end", "matched_report_date"],
            right_on=["ts_code", "period_end", "report_date"],
            how="left",
        ).drop(columns=["report_date"], errors="ignore")

    matched_events["forecast_surprise_raw"] = np.where(
        matched_events["event_type"].eq("preannouncement") | matched_events["event_type"].eq("revision"),
        matched_events["event_value_np"] - matched_events["expected_np"],
        np.nan,
    )
    matched_events["forecast_surprise_pct"] = matched_events["forecast_surprise_raw"] / matched_events["expected_np"].abs().replace(0, np.nan)

    matched_events["express_surprise_raw"] = np.where(
        matched_events["event_type"].eq("express"),
        matched_events["event_value_np"] - matched_events["expected_np"],
        np.nan,
    )
    matched_events["express_surprise_pct"] = matched_events["express_surprise_raw"] / matched_events["expected_np"].abs().replace(0, np.nan)

    matched_events["final_surprise_raw"] = np.where(
        matched_events["event_type"].eq("formal_release"),
        matched_events["event_value_np"] - matched_events["expected_np"],
        np.nan,
    )
    matched_events["final_surprise_pct"] = matched_events["final_surprise_raw"] / matched_events["expected_np"].abs().replace(0, np.nan)

    matched_events["main_surprise_raw"] = (
        matched_events["forecast_surprise_raw"]
        .fillna(matched_events["express_surprise_raw"])
        .fillna(matched_events["final_surprise_raw"])
    )
    matched_events["main_surprise_pct"] = (
        matched_events["forecast_surprise_pct"]
        .fillna(matched_events["express_surprise_pct"])
        .fillna(matched_events["final_surprise_pct"])
    )

    matched_events["surprise_family"] = np.select(
        [
            matched_events["event_type"].isin(["preannouncement", "revision"]),
            matched_events["event_type"].eq("express"),
            matched_events["event_type"].eq("formal_release"),
        ],
        ["forecast", "express", "formal_release"],
        default="other",
    )
    matched_events["main_surprise_std"] = matched_events.groupby("surprise_family")["main_surprise_pct"].transform(_robust_zscore)
    matched_events["main_surprise_std"] = matched_events["main_surprise_std"].clip(-5, 5)

    return matched_events, expectation_audit


def apply_event_filters(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    e = events_df.copy()
    p = prices_df.copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    p = p.sort_values(["ts_code", "trade_date"])
    if "vol" in p.columns:
        p["vol"] = pd.to_numeric(p["vol"], errors="coerce")

    evt_px = p[["ts_code", "trade_date", "ret", "vol"]].rename(
        columns={"trade_date": "event_trade_date", "ret": "event_day_ret", "vol": "event_day_vol"}
    )
    e = e.merge(evt_px, on=["ts_code", "event_trade_date"], how="left")
    e = e[e["event_day_ret"].notna()].copy()
    e = e[e["event_day_vol"].fillna(0) > config.min_positive_volume].copy()

    cal = pd.Series(pd.to_datetime(market_df["trade_date"].dropna().unique())).sort_values().reset_index(drop=True)
    cal_map = {d: i for i, d in enumerate(cal)}
    e["list_trade_day_idx"] = e["list_date"].apply(lambda d: _first_trade_idx_after(d, cal))
    e["event_trade_day_idx"] = e["event_trade_date"].map(cal_map)
    e["listed_days"] = e["event_trade_day_idx"] - e["list_trade_day_idx"]
    e = e[e["listed_days"] >= config.min_listed_trading_days].copy()

    if not daily_basic_df.empty:
        db = daily_basic_df.copy()
        db["trade_date"] = pd.to_datetime(db["trade_date"])
        turn = db[["ts_code", "trade_date", "turnover20"]].rename(columns={"trade_date": "event_trade_date"})
        e = e.merge(turn, on=["ts_code", "event_trade_date"], how="left")
        e = e[e["turnover20"].fillna(0) >= config.turnover20_threshold].copy()
    else:
        e["turnover20"] = np.nan

    return e.reset_index(drop=True)


def build_tushare_event_panel(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    config: ProjectConfig,
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
        db = db.sort_values(["ts_code", "trade_date"])

    rows = []
    path_rows = []
    max_window = max(config.event_windows)
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

        cars = {f"CAR{w}": post.loc[post["event_day"] <= w, "abret"].sum() for w in config.event_windows}
        beta = _estimate_beta(mm, event_date, est_window=config.beta_estimation_window)

        control = {}
        if not db.empty:
            d = db[(db["ts_code"] == ts_code) & (db["trade_date"] <= event_date)].tail(1)
            if not d.empty:
                control = {
                    "total_mv": d.get("total_mv", pd.Series([np.nan])).iloc[0],
                    "circ_mv": d.get("circ_mv", pd.Series([np.nan])).iloc[0],
                    "turnover_rate": d.get("turnover_rate", pd.Series([np.nan])).iloc[0],
                    "turnover_rate_f": d.get("turnover_rate_f", pd.Series([np.nan])).iloc[0],
                    "pe_ttm": d.get("pe_ttm", pd.Series([np.nan])).iloc[0],
                    "pb": d.get("pb", pd.Series([np.nan])).iloc[0],
                    "ps_ttm": d.get("ps_ttm", pd.Series([np.nan])).iloc[0],
                    "turnover20": d.get("turnover20", pd.Series([np.nan])).iloc[0],
                }
            else:
                control = {c: np.nan for c in ["total_mv", "circ_mv", "turnover_rate", "turnover_rate_f", "pe_ttm", "pb", "ps_ttm", "turnover20"]}
        else:
            control = {c: np.nan for c in ["total_mv", "circ_mv", "turnover_rate", "turnover_rate_f", "pe_ttm", "pb", "ps_ttm", "turnover20"]}

        row = {
            "event_id": ev.get("event_id"),
            "ts_code": ts_code,
            "industry": ev.get("industry", np.nan),
            "event_type": ev["event_type"],
            "event_subtype": ev.get("event_subtype", np.nan),
            "period_end": pd.to_datetime(ev["period_end"]).strftime("%Y-%m-%d"),
            "announcement_date": pd.to_datetime(ev["ann_date"]).strftime("%Y-%m-%d"),
            "event_trade_date": event_date.strftime("%Y-%m-%d"),
            "event_source": ev.get("event_source", np.nan),
            "expectation_source": ev.get("expectation_source", np.nan),
            "report_rc_match_quality": ev.get("report_rc_match_quality", np.nan),
            "matched_report_date": ev.get("matched_report_date", pd.NaT),
            "matched_report_count": ev.get("matched_report_count", np.nan),
            "matched_broker_count": ev.get("matched_broker_count", np.nan),
            "expected_np": ev.get("expected_np", np.nan),
            "expected_eps": ev.get("expected_eps", np.nan),
            "expected_target_price": ev.get("expected_target_price", np.nan),
            "expected_rating": ev.get("expected_rating", np.nan),
            "text_over_expectation_fraction": ev.get("text_over_expectation_fraction", np.nan),
            "main_surprise_raw": ev.get("main_surprise_raw", np.nan),
            "main_surprise_pct": ev.get("main_surprise_pct", np.nan),
            "main_surprise_std": ev.get("main_surprise_std", np.nan),
            "forecast_surprise_raw": ev.get("forecast_surprise_raw", np.nan),
            "express_surprise_raw": ev.get("express_surprise_raw", np.nan),
            "final_surprise_raw": ev.get("final_surprise_raw", np.nan),
            "revision_magnitude_np": ev.get("revision_magnitude_np", np.nan),
            "revision_magnitude_eps": ev.get("revision_magnitude_eps", np.nan),
            "target_price_change": ev.get("target_price_change", np.nan),
            "upward_revision_count": ev.get("upward_revision_count", np.nan),
            "downward_revision_count": ev.get("downward_revision_count", np.nan),
            "fraction_upgraded": ev.get("fraction_upgraded", np.nan),
            "event_value_np": ev.get("event_value_np", np.nan),
            "event_value_eps": ev.get("event_value_eps", np.nan),
            "event_value_yoy": ev.get("event_value_yoy", np.nan),
            "beta": beta,
            **control,
        }
        row["log_total_mv"] = np.log(row["total_mv"]) if pd.notna(row["total_mv"]) and row["total_mv"] > 0 else np.nan
        row["book_to_market"] = 1.0 / row["pb"] if pd.notna(row["pb"]) and row["pb"] not in (0, np.nan) else np.nan
        row.update(cars)
        rows.append(row)

        tmp = post.loc[post["event_day"] <= max_window, ["event_day", "abret"]].copy()
        tmp["event_id"] = ev.get("event_id")
        tmp["ts_code"] = ts_code
        tmp["event_trade_date"] = event_date
        tmp["main_surprise_std"] = ev.get("main_surprise_std", np.nan)
        tmp["event_type"] = ev["event_type"]
        path_rows.append(tmp)

    return pd.DataFrame(rows), (pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame())


def build_legacy_guidance_panel(
    guidance_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = build_guidance_events(guidance_df, stocks_df, market_df, logger)
    if events.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    from src.guidance_design import apply_tradability_filters, add_event_returns_and_controls

    filtered = apply_tradability_filters(
        events_df=events,
        prices_df=prices_df,
        daily_basic_df=daily_basic_df,
        market_df=market_df,
        min_listed_trading_days=120,
        turnover20_threshold=0.5,
    )
    dataset, paths = add_event_returns_and_controls(
        events_df=filtered,
        prices_df=prices_df,
        market_df=market_df,
        daily_basic_df=daily_basic_df,
        event_windows=(20, 60),
    )
    return events, dataset, paths


def _robust_zscore(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    median = clean.median()
    mad = (clean - median).abs().median()
    scale = 1.4826 * mad if pd.notna(mad) and mad > 0 else clean.std(ddof=0)
    scale = scale if pd.notna(scale) and scale > 0 else 1.0
    return (clean - median) / scale
