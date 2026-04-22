from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.expectation_alignment import build_sell_side_revision_panel, match_expectations_to_events
from src.guidance_design import _estimate_beta, _first_trade_idx_after, _next_trade_day, build_guidance_events
from src.tushare_normalization import parse_date_series


def _coerce_datetime_value(value) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    parsed = parse_date_series(pd.Series([value]))
    return parsed.iloc[0] if not parsed.empty else pd.NaT


def build_tushare_events(
    stocks_df: pd.DataFrame,
    market_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    express_df: pd.DataFrame,
    fina_df: pd.DataFrame,
    report_rc_df: pd.DataFrame,
    config: ProjectConfig,
    match_tier: str = "strict_same_quarter",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if stocks_df.empty or market_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    events = pd.concat(event_frames, ignore_index=True, sort=False)
    events = events.sort_values(["event_trade_date", "ts_code", "period_end", "ann_date"]).reset_index(drop=True)
    events["event_id"] = np.arange(1, len(events) + 1)
    events["announcement_date"] = events["ann_date"].apply(_coerce_datetime_value)
    events["period_end"] = events["period_end"].apply(_coerce_datetime_value)
    events["event_trade_date"] = events["event_trade_date"].apply(_coerce_datetime_value)

    matched_events, expectation_audit, expectation_candidates = match_expectations_to_events(
        events,
        report_rc_df,
        config,
        selected_tier=match_tier,
    )
    revision_panel = build_sell_side_revision_panel(report_rc_df)
    if not revision_panel.empty:
        matched_events = matched_events.merge(
            revision_panel,
            left_on=["ts_code", "period_end", "matched_report_date"],
            right_on=["ts_code", "period_end", "report_date"],
            how="left",
        ).drop(columns=["report_date"], errors="ignore")

    matched_events["expected_value_primary"] = np.where(
        matched_events["benchmark_value_field"].eq("eps") & matched_events["expected_eps"].notna(),
        matched_events["expected_eps"],
        matched_events["expected_np"],
    )
    matched_events["event_value_primary"] = np.where(
        matched_events["benchmark_value_field"].eq("eps") & matched_events["event_value_eps"].notna(),
        matched_events["event_value_eps"],
        matched_events["event_value_np"],
    )
    matched_events["benchmark_basis"] = np.where(
        matched_events["benchmark_value_field"].eq("eps") & matched_events["event_value_eps"].notna(),
        "eps",
        "np",
    )
    matched_events["benchmark_basis_mismatch_flag"] = (
        matched_events["expected_np"].notna()
        & matched_events["expected_eps"].isna()
        & matched_events["event_value_eps"].notna()
        & matched_events["event_value_np"].isna()
    ).astype(int)

    matched_events["forecast_surprise_raw"] = np.where(
        matched_events["event_type"].isin(["preannouncement", "revision"]),
        matched_events["event_value_primary"] - matched_events["expected_value_primary"],
        np.nan,
    )
    matched_events["forecast_surprise_pct"] = matched_events["forecast_surprise_raw"] / matched_events["expected_value_primary"].abs().replace(0, np.nan)

    matched_events["express_surprise_raw"] = np.where(
        matched_events["event_type"].eq("express"),
        matched_events["event_value_primary"] - matched_events["expected_value_primary"],
        np.nan,
    )
    matched_events["express_surprise_pct"] = matched_events["express_surprise_raw"] / matched_events["expected_value_primary"].abs().replace(0, np.nan)

    matched_events["final_surprise_raw"] = np.where(
        matched_events["event_type"].eq("formal_release"),
        matched_events["event_value_primary"] - matched_events["expected_value_primary"],
        np.nan,
    )
    matched_events["final_surprise_pct"] = matched_events["final_surprise_raw"] / matched_events["expected_value_primary"].abs().replace(0, np.nan)

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
    matched_events["main_surprise_std_event_type"] = matched_events.groupby("event_type")["main_surprise_pct"].transform(_robust_zscore)
    matched_events["main_surprise_std_event_type"] = matched_events["main_surprise_std_event_type"].clip(-5, 5)
    matched_events["usable_surprise_flag"] = matched_events["main_surprise_std"].notna().astype(int)
    matched_events["usable_raw_surprise_flag"] = matched_events["main_surprise_raw"].notna().astype(int)
    matched_events["usable_pct_surprise_flag"] = matched_events["main_surprise_pct"].notna().astype(int)
    matched_events["headline_sample_flag"] = (
        matched_events["event_type"].eq("preannouncement")
        & matched_events.get("report_rc_match_tier", pd.Series(index=matched_events.index, dtype=object)).eq("strict_same_quarter")
    ).astype(int)

    return matched_events, expectation_audit, expectation_candidates


def annotate_event_filters(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()

    e = events_df.copy()
    e["event_trade_date"] = pd.to_datetime(e["event_trade_date"], errors="coerce")
    e["list_date"] = e["list_date"].apply(_coerce_datetime_value)
    e["is_non_st_name"] = ~e.get("name", pd.Series(index=e.index, dtype=object)).fillna("").str.contains("ST", case=False, regex=False)

    p = prices_df.copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    p = p.sort_values(["ts_code", "trade_date"])
    p["ret"] = pd.to_numeric(p["ret"], errors="coerce")
    if "vol" in p.columns:
        p["vol"] = pd.to_numeric(p["vol"], errors="coerce")
    else:
        p["vol"] = np.nan

    evt_px = p[["ts_code", "trade_date", "ret", "vol"]].rename(
        columns={"trade_date": "event_trade_date", "ret": "event_day_ret", "vol": "event_day_vol"}
    )
    e = e.merge(evt_px, on=["ts_code", "event_trade_date"], how="left")
    e["has_event_day_ret"] = e["event_day_ret"].notna()
    e["has_positive_event_day_volume"] = e["event_day_vol"].fillna(0) > config.min_positive_volume

    cal = pd.Series(pd.to_datetime(market_df["trade_date"].dropna().unique(), errors="coerce")).dropna().sort_values().reset_index(drop=True)
    cal_map = {d: i for i, d in enumerate(cal)}
    e["list_trade_day_idx"] = e["list_date"].apply(lambda d: _first_trade_idx_after(d, cal))
    e["event_trade_day_idx"] = e["event_trade_date"].map(cal_map)
    e["listed_days"] = e["event_trade_day_idx"] - e["list_trade_day_idx"]
    e["passes_listing_age"] = e["listed_days"] >= config.min_listed_trading_days

    if not daily_basic_df.empty:
        db = daily_basic_df.copy()
        db["trade_date"] = pd.to_datetime(db["trade_date"], errors="coerce")
        turn = db[["ts_code", "trade_date", "turnover20"]].rename(columns={"trade_date": "event_trade_date"})
        e = e.merge(turn, on=["ts_code", "event_trade_date"], how="left")
    else:
        e["turnover20"] = np.nan

    e["passes_turnover_threshold"] = e["turnover20"].fillna(0) >= config.turnover20_threshold
    e["has_expectation_match"] = e["expected_value_primary"].notna()
    e["has_usable_standardized_surprise"] = e["main_surprise_std"].notna()
    e["has_usable_raw_surprise"] = e["main_surprise_raw"].notna()
    e["has_usable_pct_surprise"] = e["main_surprise_pct"].notna()
    e["is_strict_match_tier"] = e.get("report_rc_match_tier", pd.Series(index=e.index, dtype=object)).eq("strict_same_quarter")
    e["headline_sample_flag"] = e.get("headline_sample_flag", pd.Series(index=e.index, dtype=float)).fillna(0).astype(int)
    e["passes_restrictive_filters"] = (
        e["is_non_st_name"]
        & e["has_event_day_ret"]
        & e["has_positive_event_day_volume"]
        & e["passes_listing_age"]
        & e["passes_turnover_threshold"]
    )
    e["passes_relaxed_filters"] = (
        e["is_non_st_name"]
        & e["has_event_day_ret"]
        & e["passes_listing_age"]
    )
    return e.reset_index(drop=True)


def apply_event_filters(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    market_df: pd.DataFrame,
    config: ProjectConfig,
    profile: str = "restrictive",
) -> pd.DataFrame:
    annotated = annotate_event_filters(
        events_df=events_df,
        prices_df=prices_df,
        daily_basic_df=daily_basic_df,
        market_df=market_df,
        config=config,
    )
    if annotated.empty:
        return annotated
    return apply_filter_profile(annotated, profile)


def build_filter_funnel(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["stage", "event_count"])
    stages = [
        ("all_events", pd.Series(True, index=events_df.index)),
        ("non_st", events_df.get("is_non_st_name", pd.Series(False, index=events_df.index))),
        (
            "has_event_day_ret",
            events_df.get("is_non_st_name", pd.Series(False, index=events_df.index))
            & events_df.get("has_event_day_ret", pd.Series(False, index=events_df.index)),
        ),
        (
            "positive_event_day_volume",
            events_df.get("is_non_st_name", pd.Series(False, index=events_df.index))
            & events_df.get("has_event_day_ret", pd.Series(False, index=events_df.index))
            & events_df.get("has_positive_event_day_volume", pd.Series(False, index=events_df.index)),
        ),
        (
            "listing_age",
            events_df.get("is_non_st_name", pd.Series(False, index=events_df.index))
            & events_df.get("has_event_day_ret", pd.Series(False, index=events_df.index))
            & events_df.get("has_positive_event_day_volume", pd.Series(False, index=events_df.index))
            & events_df.get("passes_listing_age", pd.Series(False, index=events_df.index)),
        ),
        ("turnover_threshold", events_df.get("passes_restrictive_filters", pd.Series(False, index=events_df.index))),
        (
            "expectation_match",
            events_df.get("passes_restrictive_filters", pd.Series(False, index=events_df.index))
            & events_df.get("has_expectation_match", pd.Series(False, index=events_df.index)),
        ),
        (
            "usable_standardized_surprise",
            events_df.get("passes_restrictive_filters", pd.Series(False, index=events_df.index))
            & events_df.get("has_usable_standardized_surprise", pd.Series(False, index=events_df.index)),
        ),
    ]
    return pd.DataFrame(
        [{"stage": stage, "event_count": int(mask.fillna(False).sum())} for stage, mask in stages]
    )


def summarize_window_availability(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame(columns=["window", "available_events"])

    p = prices_df.copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    p = p.sort_values(["ts_code", "trade_date"])

    rows: list[dict[str, object]] = []
    for window in sorted(config.event_windows):
        available = 0
        for _, ev in events_df.iterrows():
            ts_code = ev["ts_code"]
            event_date = pd.to_datetime(ev["event_trade_date"], errors="coerce")
            post = p[(p["ts_code"] == ts_code) & (p["trade_date"] >= event_date)]
            if len(post) >= window:
                available += 1
        rows.append({"window": f"CAR{window}", "available_events": available})
    return pd.DataFrame(rows)


def apply_filter_profile(events_df: pd.DataFrame, profile: str) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    flag_col = _profile_flag(profile)
    return events_df[events_df[flag_col].fillna(False)].reset_index(drop=True)


def _window_availability_for_event(mm: pd.DataFrame, event_date: pd.Timestamp, windows: tuple[int, ...]) -> dict[int, bool]:
    post = mm[mm["trade_date"] >= event_date].copy().reset_index(drop=True)
    return {window: len(post) >= window for window in windows}


def _profile_flag(profile: str) -> str:
    return "passes_relaxed_filters" if (profile or "restrictive").strip().lower() == "relaxed" else "passes_restrictive_filters"


def _signal_column(signal_scale: str, event_standardized: bool = False) -> str:
    normalized = (signal_scale or "").strip().lower()
    if normalized == "raw":
        return "main_surprise_raw"
    if normalized == "pct":
        return "main_surprise_pct"
    return "main_surprise_std_event_type" if event_standardized else "main_surprise_std"


def build_benchmark_quality_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    summary = (
        events_df.groupby(["benchmark_method", "report_rc_match_tier", "report_rc_match_tier_group", "event_type"], dropna=False, as_index=False)
        .agg(
            event_count=("event_id", "count"),
            strict_count=("report_rc_match_quality", lambda s: int((s == "strict").sum())),
            usable_signal_count=("main_surprise_std", lambda s: int(s.notna().sum())),
            usable_raw_signal_count=("main_surprise_raw", lambda s: int(s.notna().sum())),
            usable_pct_signal_count=("main_surprise_pct", lambda s: int(s.notna().sum())),
            median_report_count=("matched_report_count", "median"),
            median_broker_count=("matched_broker_count", "median"),
            median_lag_days=("benchmark_lag_days", "median"),
        )
        .sort_values(["benchmark_method", "event_type"])
        .reset_index(drop=True)
    )
    summary["strict_share"] = summary["strict_count"] / summary["event_count"].replace(0, np.nan)
    summary["usable_signal_share"] = summary["usable_signal_count"] / summary["event_count"].replace(0, np.nan)
    return summary


def build_timing_alignment_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    rows = []
    for (match_tier, event_type), subset in events_df.groupby(["report_rc_match_tier", "event_type"], dropna=False):
        rows.append(
            {
                "report_rc_match_tier": match_tier,
                "report_rc_match_tier_group": subset.get("report_rc_match_tier_group", pd.Series(index=subset.index, dtype=object)).iloc[0] if not subset.empty else np.nan,
                "event_type": event_type,
                "event_count": int(len(subset)),
                "matched_count": int(subset["matched_report_date"].notna().sum()),
                "median_lag_days": float(subset["benchmark_lag_days"].median()) if subset["benchmark_lag_days"].notna().any() else np.nan,
                "p90_lag_days": float(subset["benchmark_lag_days"].quantile(0.9)) if subset["benchmark_lag_days"].notna().any() else np.nan,
                "same_day_or_later_violations": int((subset["benchmark_lag_days"].fillna(1) <= 0).sum()),
                "basis_mismatch_count": int(subset.get("benchmark_basis_mismatch_flag", pd.Series(0, index=subset.index)).fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_event_type_signal_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    rows = []
    for (match_tier, event_type), subset in events_df.groupby(["report_rc_match_tier", "event_type"], dropna=False):
        rows.append(
            {
                "report_rc_match_tier": match_tier,
                "report_rc_match_tier_group": subset.get("report_rc_match_tier_group", pd.Series(index=subset.index, dtype=object)).iloc[0] if not subset.empty else np.nan,
                "event_type": event_type,
                "event_count": int(len(subset)),
                "usable_raw_signal_count": int(subset["main_surprise_raw"].notna().sum()),
                "usable_pct_signal_count": int(subset["main_surprise_pct"].notna().sum()),
                "usable_std_signal_count": int(subset["main_surprise_std"].notna().sum()),
                "median_abs_raw_surprise": float(subset["main_surprise_raw"].abs().median()) if subset["main_surprise_raw"].notna().any() else np.nan,
                "median_abs_pct_surprise": float(subset["main_surprise_pct"].abs().median()) if subset["main_surprise_pct"].notna().any() else np.nan,
                "median_abs_std_surprise": float(subset["main_surprise_std"].abs().median()) if subset["main_surprise_std"].notna().any() else np.nan,
                "median_broker_count": float(subset["matched_broker_count"].median()) if subset["matched_broker_count"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_expectation_coverage_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    df = events_df.copy()
    df["year"] = pd.to_datetime(df["event_trade_date"], errors="coerce").dt.year
    summary = (
        df.groupby(["year", "report_rc_match_tier", "report_rc_match_tier_group", "event_type"], dropna=False, as_index=False)
        .agg(
            event_count=("event_id", "count"),
            expectation_match_count=("has_expectation_match", lambda s: int(s.fillna(False).sum())),
            usable_signal_count=("has_usable_standardized_surprise", lambda s: int(s.fillna(False).sum())),
            strict_count=("report_rc_match_quality", lambda s: int((s == "strict").sum())),
        )
        .sort_values(["year", "event_type"])
        .reset_index(drop=True)
    )
    summary["expectation_match_share"] = summary["expectation_match_count"] / summary["event_count"].replace(0, np.nan)
    summary["usable_signal_share"] = summary["usable_signal_count"] / summary["event_count"].replace(0, np.nan)
    summary["strict_share"] = summary["strict_count"] / summary["event_count"].replace(0, np.nan)
    return summary


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
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    p = p.sort_values(["ts_code", "trade_date"])
    p["ret"] = pd.to_numeric(p["ret"], errors="coerce")

    m = market_df.copy()
    m["trade_date"] = pd.to_datetime(m["trade_date"], errors="coerce")
    m["mkt_ret"] = pd.to_numeric(m["mkt_ret"], errors="coerce")
    m = m.dropna(subset=["trade_date", "mkt_ret"]).sort_values("trade_date")

    db = daily_basic_df.copy()
    if not db.empty:
        db["trade_date"] = pd.to_datetime(db["trade_date"], errors="coerce")
        db = db.sort_values(["ts_code", "trade_date"])

    rows = []
    path_rows = []
    max_window = max(config.event_windows)
    windows = tuple(sorted(config.event_windows))

    for _, ev in events_df.iterrows():
        ts_code = ev["ts_code"]
        event_date = pd.to_datetime(ev["event_trade_date"], errors="coerce")
        sp = p[p["ts_code"] == ts_code][["trade_date", "ret"]].dropna()
        if sp.empty or pd.isna(event_date):
            continue

        mm = sp.merge(m[["trade_date", "mkt_ret"]], on="trade_date", how="inner")
        mm = mm.sort_values("trade_date")
        mm["abret"] = mm["ret"] - mm["mkt_ret"]
        post = mm[mm["trade_date"] >= event_date].copy().reset_index(drop=True)
        if post.empty:
            continue
        post["event_day"] = np.arange(1, len(post) + 1)

        availability = _window_availability_for_event(mm, event_date, windows)
        cars = {
            f"CAR{window}": post.loc[post["event_day"] <= window, "abret"].sum() if availability[window] else np.nan
            for window in windows
        }
        beta = _estimate_beta(mm, event_date, est_window=config.beta_estimation_window)

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
            "event_type": ev.get("event_type", np.nan),
            "event_subtype": ev.get("event_subtype", np.nan),
            "period_end": pd.to_datetime(ev.get("period_end"), errors="coerce").strftime("%Y-%m-%d") if pd.notna(ev.get("period_end")) else np.nan,
            "announcement_date": pd.to_datetime(ev.get("ann_date"), errors="coerce").strftime("%Y-%m-%d") if pd.notna(ev.get("ann_date")) else np.nan,
            "event_trade_date": event_date.strftime("%Y-%m-%d"),
            "event_source": ev.get("event_source", np.nan),
            "expectation_source": ev.get("expectation_source", np.nan),
            "benchmark_method": ev.get("benchmark_method", np.nan),
            "benchmark_value_field": ev.get("benchmark_value_field", np.nan),
            "benchmark_basis": ev.get("benchmark_basis", np.nan),
            "benchmark_basis_mismatch_flag": ev.get("benchmark_basis_mismatch_flag", np.nan),
            "report_rc_match_quality": ev.get("report_rc_match_quality", np.nan),
            "report_rc_match_tier": ev.get("report_rc_match_tier", np.nan),
            "report_rc_match_tier_group": ev.get("report_rc_match_tier_group", np.nan),
            "matched_report_date": ev.get("matched_report_date", pd.NaT),
            "matched_report_count": ev.get("matched_report_count", np.nan),
            "matched_broker_count": ev.get("matched_broker_count", np.nan),
            "benchmark_lag_days": ev.get("benchmark_lag_days", np.nan),
            "candidate_report_count_total": ev.get("candidate_report_count_total", np.nan),
            "candidate_broker_count_total": ev.get("candidate_broker_count_total", np.nan),
            "selected_candidate_rows": ev.get("selected_candidate_rows", np.nan),
            "benchmark_value_count": ev.get("benchmark_value_count", np.nan),
            "benchmark_has_np": ev.get("benchmark_has_np", np.nan),
            "benchmark_has_eps": ev.get("benchmark_has_eps", np.nan),
            "expected_np": ev.get("expected_np", np.nan),
            "expected_eps": ev.get("expected_eps", np.nan),
            "expected_target_price": ev.get("expected_target_price", np.nan),
            "expected_rating": ev.get("expected_rating", np.nan),
            "expected_value_primary": ev.get("expected_value_primary", np.nan),
            "text_over_expectation_fraction": ev.get("text_over_expectation_fraction", np.nan),
            "main_surprise_raw": ev.get("main_surprise_raw", np.nan),
            "main_surprise_pct": ev.get("main_surprise_pct", np.nan),
            "main_surprise_std": ev.get("main_surprise_std", np.nan),
            "main_surprise_std_event_type": ev.get("main_surprise_std_event_type", np.nan),
            "forecast_surprise_raw": ev.get("forecast_surprise_raw", np.nan),
            "forecast_surprise_pct": ev.get("forecast_surprise_pct", np.nan),
            "express_surprise_raw": ev.get("express_surprise_raw", np.nan),
            "express_surprise_pct": ev.get("express_surprise_pct", np.nan),
            "final_surprise_raw": ev.get("final_surprise_raw", np.nan),
            "final_surprise_pct": ev.get("final_surprise_pct", np.nan),
            "surprise_family": ev.get("surprise_family", np.nan),
            "usable_surprise_flag": ev.get("usable_surprise_flag", np.nan),
            "usable_raw_surprise_flag": ev.get("usable_raw_surprise_flag", np.nan),
            "usable_pct_surprise_flag": ev.get("usable_pct_surprise_flag", np.nan),
            "headline_sample_flag": ev.get("headline_sample_flag", np.nan),
            "revision_magnitude_np": ev.get("revision_magnitude_np", np.nan),
            "revision_magnitude_eps": ev.get("revision_magnitude_eps", np.nan),
            "target_price_change": ev.get("target_price_change", np.nan),
            "upward_revision_count": ev.get("upward_revision_count", np.nan),
            "downward_revision_count": ev.get("downward_revision_count", np.nan),
            "fraction_upgraded": ev.get("fraction_upgraded", np.nan),
            "event_value_np": ev.get("event_value_np", np.nan),
            "event_value_eps": ev.get("event_value_eps", np.nan),
            "event_value_yoy": ev.get("event_value_yoy", np.nan),
            "event_value_primary": ev.get("event_value_primary", np.nan),
            "is_non_st_name": ev.get("is_non_st_name", np.nan),
            "has_event_day_ret": ev.get("has_event_day_ret", np.nan),
            "has_positive_event_day_volume": ev.get("has_positive_event_day_volume", np.nan),
            "passes_listing_age": ev.get("passes_listing_age", np.nan),
            "passes_turnover_threshold": ev.get("passes_turnover_threshold", np.nan),
            "has_expectation_match": ev.get("has_expectation_match", np.nan),
            "has_usable_standardized_surprise": ev.get("has_usable_standardized_surprise", np.nan),
            "has_usable_raw_surprise": ev.get("has_usable_raw_surprise", np.nan),
            "passes_restrictive_filters": ev.get("passes_restrictive_filters", np.nan),
            "passes_relaxed_filters": ev.get("passes_relaxed_filters", np.nan),
            "listed_days": ev.get("listed_days", np.nan),
            "beta": beta,
            **control,
        }
        row["log_total_mv"] = np.log(row["total_mv"]) if pd.notna(row["total_mv"]) and row["total_mv"] > 0 else np.nan
        row["book_to_market"] = 1.0 / row["pb"] if pd.notna(row["pb"]) and row["pb"] != 0 else np.nan
        row["available_post_event_days"] = int(len(post))
        for window, is_available in availability.items():
            row[f"CAR{window}_available"] = int(is_available)
        row.update(cars)
        rows.append(row)

        tmp = post.loc[post["event_day"] <= max_window, ["trade_date", "event_day", "abret"]].copy()
        tmp["event_id"] = ev.get("event_id")
        tmp["ts_code"] = ts_code
        tmp["event_trade_date"] = event_date
        tmp["main_surprise_std"] = ev.get("main_surprise_std", np.nan)
        tmp["main_surprise_std_event_type"] = ev.get("main_surprise_std_event_type", np.nan)
        tmp["main_surprise_raw"] = ev.get("main_surprise_raw", np.nan)
        tmp["main_surprise_pct"] = ev.get("main_surprise_pct", np.nan)
        tmp["report_rc_match_tier"] = ev.get("report_rc_match_tier", np.nan)
        tmp["report_rc_match_tier_group"] = ev.get("report_rc_match_tier_group", np.nan)
        tmp["headline_sample_flag"] = ev.get("headline_sample_flag", np.nan)
        tmp["event_type"] = ev.get("event_type", np.nan)
        path_rows.append(tmp)

    panel = pd.DataFrame(rows)
    paths = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    return panel, paths


def build_annotated_event_panel(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_tushare_event_panel(
        events_df=events_df,
        prices_df=prices_df,
        market_df=market_df,
        daily_basic_df=daily_basic_df,
        config=config,
    )


def build_ablation_panel(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    config: ProjectConfig,
    profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = apply_filter_profile(events_df, profile)
    return build_tushare_event_panel(
        events_df=filtered,
        prices_df=prices_df,
        market_df=market_df,
        daily_basic_df=daily_basic_df,
        config=config,
    )


def list_ablation_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    spec_id = 1
    for benchmark_method in ["latest_snapshot", "latest_per_analyst", "pooled_median"]:
        for event_universe in ["preannouncement_only", "all_event_types"]:
            for signal_scale in ["raw", "pct", "std"]:
                for match_tier in [
                    "strict_same_quarter",
                    "same_fiscal_year_nearest_valid",
                    "latest_valid_pre_event",
                    "multi_report_median",
                ]:
                    for analyst_min in [1, 2, 3, 5]:
                        for profile in ["restrictive", "relaxed"]:
                            specs.append(
                                {
                                    "spec_id": f"spec_{spec_id:03d}",
                                    "benchmark_method": benchmark_method,
                                    "event_universe": event_universe,
                                    "signal_scale": signal_scale,
                                    "match_tier": match_tier,
                                    "match_tier_group": "strict" if match_tier == "strict_same_quarter" else "relaxed",
                                    "analyst_min": analyst_min,
                                    "filter_profile": profile,
                                }
                            )
                            spec_id += 1
    return specs


def apply_ablation_spec(events_df: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    df = events_df.copy()
    df = df[df["benchmark_method"].eq(spec["benchmark_method"])]
    df = df[df.get("report_rc_match_tier", pd.Series(index=df.index, dtype=object)).eq(spec["match_tier"])]
    if spec["event_universe"] == "preannouncement_only":
        df = df[df["event_type"].eq("preannouncement")]
    df = df[df["matched_broker_count"].fillna(0) >= int(spec["analyst_min"])]
    df = apply_filter_profile(df, str(spec["filter_profile"]))
    signal_col = _signal_column(str(spec["signal_scale"]))
    if signal_col not in df.columns:
        return pd.DataFrame()
    df = df[df[signal_col].notna()].copy()
    df["active_signal_col"] = signal_col
    return df.reset_index(drop=True)


def choose_strongest_spec(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    scored = results_df.copy()
    scored["usable"] = (
        scored["regression_nobs"].fillna(0) >= 60
    ) & scored["coef"].notna() & scored["p_value"].notna()
    scored = scored[scored["usable"]].copy()
    if scored.empty:
        return pd.DataFrame()
    scored["benchmark_rank"] = scored["benchmark_method"].map({"latest_per_analyst": 0, "latest_snapshot": 1, "pooled_median": 2}).fillna(9)
    scored["event_rank"] = scored["event_universe"].map({"preannouncement_only": 0, "all_event_types": 1}).fillna(9)
    scored["match_tier_rank"] = scored["match_tier"].map({
        "strict_same_quarter": 0,
        "same_fiscal_year_nearest_valid": 1,
        "latest_valid_pre_event": 2,
        "multi_report_median": 3,
    }).fillna(9)
    scored["signal_rank"] = scored["signal_scale"].map({"raw": 0, "pct": 1, "std": 2}).fillna(9)
    scored = scored.sort_values(
        ["benchmark_rank", "event_rank", "match_tier_rank", "signal_rank", "analyst_min", "car_window", "p_value", "regression_nobs"],
        ascending=[True, True, True, True, True, True, True, False],
    )
    return scored.head(1).reset_index(drop=True)


def recommendation_from_diagnostics(
    strongest_spec_df: pd.DataFrame,
    failure_df: pd.DataFrame,
) -> tuple[str, str]:
    if strongest_spec_df.empty:
        return "B", "No specification survives credibly with sufficient sample and stable benchmark quality; keep Tushare as the baseline diagnostic layer and add stronger external expectation data later."
    top = strongest_spec_df.iloc[0]
    if (
        float(top.get("p_value", np.nan)) <= 0.10
        and int(top.get("regression_nobs", 0)) >= 80
        and top.get("event_universe") == "preannouncement_only"
        and top.get("match_tier") == "strict_same_quarter"
    ):
        return "A", "Continue with Tushare only, but keep the headline on the preannouncement-only strict-match subset rather than the pooled all-event specification."
    if not failure_df.empty and (failure_df["fixability"] == "external_required").any():
        return "B", "Keep Tushare as the baseline but add stronger external analyst-expectation data for a credible main headline."
    return "C", "Abandon the current pooled headline specification and reframe the project around a narrower question or subset where the diagnostics show cleaner identification."


def build_failure_analysis(
    event_panel_df: pd.DataFrame,
    spec_results_df: pd.DataFrame,
    filter_funnel_df: pd.DataFrame,
    benchmark_quality_df: pd.DataFrame,
) -> pd.DataFrame:
    if event_panel_df.empty:
        return pd.DataFrame()
    total_events = len(event_panel_df)
    usable_signal = int(event_panel_df["main_surprise_std"].notna().sum())
    strict_matches = int((event_panel_df["report_rc_match_quality"] == "strict").sum())
    rows = [
        {
            "rank": 1,
            "failure_reason": "poor_coverage_of_usable_surprise_rows",
            "why_it_weakens_signal": "The regression can only use a small subset of the filtered panel once benchmark-matched surprise and strict match tiers are required.",
            "observed_metric": f"usable standardized surprise rows = {usable_signal} / {total_events}",
            "likely_importance": "very_high",
            "fixability": "tushare_only",
        },
        {
            "rank": 2,
            "failure_reason": "match_tier_relaxation_changes_inference",
            "why_it_weakens_signal": "Inference depends materially on whether the benchmark uses strict same-quarter matching or looser fallback tiers.",
            "observed_metric": f"strict tier rows = {int(event_panel_df.get('report_rc_match_tier', pd.Series(index=event_panel_df.index, dtype=object)).eq('strict_same_quarter').sum())} / {total_events}",
            "likely_importance": "very_high",
            "fixability": "tushare_only",
        },
        {
            "rank": 3,
            "failure_reason": "insufficient_analyst_coverage_per_event",
            "why_it_weakens_signal": "Thin broker coverage makes the expectation benchmark noisy and unstable across events.",
            "observed_metric": f"strict report_rc matches = {strict_matches} / {total_events}",
            "likely_importance": "high",
            "fixability": "mixed",
        },
        {
            "rank": 4,
            "failure_reason": "mixed_event_types",
            "why_it_weakens_signal": "Pooling preannouncements, revisions, express releases, and formal releases can average away economically different responses.",
            "observed_metric": f"event types observed = {', '.join(sorted(event_panel_df['event_type'].dropna().astype(str).unique()))}",
            "likely_importance": "high",
            "fixability": "tushare_only",
        },
        {
            "rank": 5,
            "failure_reason": "weak_or_misaligned_expectation_benchmark",
            "why_it_weakens_signal": "report_rc expectations may not line up cleanly with event-level realized NP/EPS fields, especially for formal releases.",
            "observed_metric": f"median benchmark lag days = {event_panel_df['benchmark_lag_days'].median() if event_panel_df['benchmark_lag_days'].notna().any() else np.nan}",
            "likely_importance": "high",
            "fixability": "external_required",
        },
        {
            "rank": 6,
            "failure_reason": "standardization_and_filtering_choices",
            "why_it_weakens_signal": "Family-level standardization and restrictive turnover filters may remove or compress valid short-window signal.",
            "observed_metric": f"restrictive funnel terminal count = {int(filter_funnel_df['event_count'].iloc[-1]) if not filter_funnel_df.empty else 0}",
            "likely_importance": "medium",
            "fixability": "tushare_only",
        },
    ]
    return pd.DataFrame(rows)


def summarize_spec_event_types(df: pd.DataFrame) -> str:
    if df.empty or "event_type" not in df.columns:
        return ""
    counts = df["event_type"].value_counts().sort_index()
    return "; ".join(f"{idx}:{val}" for idx, val in counts.items())


def _regression_frame_for_spec(df: pd.DataFrame, dep: str, signal_col: str) -> pd.DataFrame:
    regressors = [signal_col, "log_total_mv", "beta", "book_to_market", "turnover20", "pe_ttm", "ps_ttm"]
    cols = [dep, "ts_code", "industry", "year_quarter", "event_type"] + regressors
    dd = df[[c for c in cols if c in df.columns]].copy()
    for col in [dep] + [c for c in regressors if c in dd.columns]:
        dd[col] = pd.to_numeric(dd[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    dd = dd.dropna(subset=[dep, signal_col, "ts_code", "industry", "year_quarter", "event_type"])
    active_regs = [c for c in regressors if c in dd.columns and dd[c].notna().sum() >= 20]
    if not active_regs or len(dd) < 30:
        return pd.DataFrame()
    return dd[[dep, "ts_code", "industry", "year_quarter", "event_type"] + active_regs].dropna().copy()


def run_spec_regression(df: pd.DataFrame, dep: str, signal_col: str) -> dict[str, object]:
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception:
        return {"coef": np.nan, "t_stat": np.nan, "p_value": np.nan, "regression_nobs": 0, "r2": np.nan}

    reg_df = _regression_frame_for_spec(df, dep, signal_col)
    if reg_df.empty:
        return {"coef": np.nan, "t_stat": np.nan, "p_value": np.nan, "regression_nobs": 0, "r2": np.nan}

    regressors = [c for c in [signal_col, "log_total_mv", "beta", "book_to_market", "turnover20", "pe_ttm", "ps_ttm"] if c in reg_df.columns]
    formula = f"{dep} ~ {' + '.join(regressors)} + C(industry) + C(year_quarter) + C(event_type)"
    model = smf.ols(formula=formula, data=reg_df).fit(cov_type="cluster", cov_kwds={"groups": reg_df["ts_code"]})
    return {
        "coef": model.params.get(signal_col, np.nan),
        "t_stat": model.tvalues.get(signal_col, np.nan),
        "p_value": model.pvalues.get(signal_col, np.nan),
        "regression_nobs": int(model.nobs),
        "r2": model.rsquared,
    }


def run_ablation_specs(event_panel_df: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if event_panel_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    specs = list_ablation_specs()
    catalog_rows = []
    result_rows = []
    df = event_panel_df.copy()
    df["event_trade_date"] = pd.to_datetime(df["event_trade_date"], errors="coerce")
    df["year_quarter"] = df["event_trade_date"].dt.to_period("Q").astype(str)
    df["industry"] = df["industry"].fillna("unknown")

    for spec in specs:
        spec_df = apply_ablation_spec(df, spec)
        signal_col = _signal_column(str(spec["signal_scale"]))
        catalog_rows.append(
            {
                **spec,
                "signal_col": signal_col,
                "event_count": int(len(spec_df)),
                "usable_signal_count": int(spec_df[signal_col].notna().sum()) if not spec_df.empty and signal_col in spec_df.columns else 0,
                "event_type_composition": summarize_spec_event_types(spec_df),
            }
        )
        for window in [3, 5, 10, 20]:
            dep = f"CAR{window}"
            if dep not in spec_df.columns:
                continue
            run_df = spec_df[spec_df[dep].notna()].copy()
            stats = run_spec_regression(run_df, dep=dep, signal_col=signal_col)
            result_rows.append(
                {
                    **spec,
                    "car_window": dep,
                    "signal_col": signal_col,
                    "event_count": int(len(run_df)),
                    "usable_signal_count": int(run_df[signal_col].notna().sum()) if signal_col in run_df.columns else 0,
                    "strict_match_share": float((run_df["report_rc_match_quality"] == "strict").mean()) if not run_df.empty else np.nan,
                    "median_broker_count": float(run_df["matched_broker_count"].median()) if not run_df.empty else np.nan,
                    "median_benchmark_lag_days": float(run_df["benchmark_lag_days"].median()) if not run_df.empty else np.nan,
                    "event_type_composition": summarize_spec_event_types(run_df),
                    **stats,
                }
            )
    return pd.DataFrame(catalog_rows), pd.DataFrame(result_rows)


def build_recommendation_note(recommendation_code: str, recommendation_text: str, strongest_spec_df: pd.DataFrame) -> str:
    lines = [
        "Tushare-first diagnostic recommendation",
        f"Recommendation: {recommendation_code}",
        recommendation_text,
    ]
    if not strongest_spec_df.empty:
        row = strongest_spec_df.iloc[0]
        lines.extend(
            [
                f"Strongest surviving specification: {row.get('spec_id')}",
                f"Benchmark method: {row.get('benchmark_method')}",
                f"Event universe: {row.get('event_universe')}",
                f"Signal scale: {row.get('signal_scale')}",
                f"Window: {row.get('car_window')}",
                f"Regression nobs: {row.get('regression_nobs')}",
                f"Coefficient: {row.get('coef')}",
                f"p-value: {row.get('p_value')}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_ablation_summary_for_run(spec_results_df: pd.DataFrame) -> dict[str, object]:
    strongest = choose_strongest_spec(spec_results_df)
    if strongest.empty:
        return {
            "strongest_spec_id": "",
            "strongest_spec_window": "",
            "strongest_spec_coef": np.nan,
            "strongest_spec_p_value": np.nan,
            "strongest_spec_nobs": 0,
        }
    row = strongest.iloc[0]
    return {
        "strongest_spec_id": row.get("spec_id", ""),
        "strongest_spec_window": row.get("car_window", ""),
        "strongest_spec_coef": row.get("coef", np.nan),
        "strongest_spec_p_value": row.get("p_value", np.nan),
        "strongest_spec_nobs": row.get("regression_nobs", 0),
    }


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
    from src.guidance_design import add_event_returns_and_controls, apply_tradability_filters

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
