from __future__ import annotations

import numpy as np
import pandas as pd


def parse_date_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    out = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    need_fallback = out.isna()
    if need_fallback.any():
        out.loc[need_fallback] = pd.to_datetime(s.loc[need_fallback], errors="coerce")
    return out


def normalize_report_rc(report_rc_df: pd.DataFrame) -> pd.DataFrame:
    if report_rc_df.empty:
        return pd.DataFrame()
    df = report_rc_df.copy()
    if "report_date" in df.columns:
        df["report_date"] = parse_date_series(df["report_date"])
    df["quarter"] = df.get("quarter", pd.Series(index=df.index, dtype=object)).astype(str).str.strip()
    q = df["quarter"].str.extract(r"(?P<year>\d{4})Q(?P<q>[1-4])")
    df["fiscal_year"] = pd.to_numeric(q["year"], errors="coerce")
    df["fiscal_quarter"] = pd.to_numeric(q["q"], errors="coerce")
    quarter_month = df["fiscal_quarter"].map({1: 3, 2: 6, 3: 9, 4: 12})
    df["period_end"] = pd.to_datetime(
        dict(year=df["fiscal_year"], month=quarter_month, day=1),
        errors="coerce",
    ) + pd.offsets.MonthEnd(0)
    numeric_cols = ["np", "eps", "pe", "max_price", "min_price", "tp"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target_price_mid"] = np.where(
        df["max_price"].notna() & df["min_price"].notna(),
        (df["max_price"] + df["min_price"]) / 2.0,
        df["tp"],
    )
    df["report_title"] = df.get("report_title", "").fillna("").astype(str)
    df["title_over_expectation_flag"] = df["report_title"].str.contains(
        "超预期|业绩超预期|利润超预期|盈利超预期", regex=True
    ).astype(int)
    return df.dropna(subset=["ts_code", "report_date"]).sort_values(
        ["ts_code", "period_end", "report_date"]
    ).reset_index(drop=True)


def normalize_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
    if forecast_df.empty:
        return pd.DataFrame()
    df = forecast_df.copy()
    for col in ["ann_date", "end_date", "first_ann_date"]:
        if col in df.columns:
            df[col] = parse_date_series(df[col])
    num_cols = [
        "p_change_min",
        "p_change_max",
        "net_profit_min",
        "net_profit_max",
        "last_parent_net",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["guidance_yoy_midpoint"] = np.where(
        df["p_change_min"].notna() | df["p_change_max"].notna(),
        (df["p_change_min"].fillna(df["p_change_max"]) + df["p_change_max"].fillna(df["p_change_min"])) / 2.0,
        np.nan,
    )
    df["forecast_profit_mid"] = np.where(
        df["net_profit_min"].notna() | df["net_profit_max"].notna(),
        (df["net_profit_min"].fillna(df["net_profit_max"]) + df["net_profit_max"].fillna(df["net_profit_min"])) / 2.0,
        np.nan,
    )
    df = df.sort_values(["ts_code", "end_date", "ann_date", "update_flag"]).reset_index(drop=True)
    df["prior_forecast_profit_mid"] = df.groupby(["ts_code", "end_date"])["forecast_profit_mid"].shift(1)
    df["prior_yoy_midpoint"] = df.groupby(["ts_code", "end_date"])["guidance_yoy_midpoint"].shift(1)
    df["revision_magnitude"] = df["forecast_profit_mid"] - df["prior_forecast_profit_mid"]
    df["is_revision"] = df.groupby(["ts_code", "end_date"]).cumcount() > 0
    df["revision_direction"] = np.select(
        [df["revision_magnitude"] > 0, df["revision_magnitude"] < 0],
        ["up", "down"],
        default="none",
    )
    return df.dropna(subset=["ts_code", "ann_date", "end_date"]).reset_index(drop=True)


def normalize_express(express_df: pd.DataFrame) -> pd.DataFrame:
    if express_df.empty:
        return pd.DataFrame()
    df = express_df.copy()
    for col in ["ann_date", "end_date"]:
        if col in df.columns:
            df[col] = parse_date_series(df[col])
    num_cols = [
        "revenue",
        "operate_profit",
        "total_profit",
        "n_income",
        "diluted_eps",
        "diluted_roe",
        "yoy_net_profit",
        "bps",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["ts_code", "end_date", "ann_date", "update_flag"]).drop_duplicates(
        subset=["ts_code", "end_date", "ann_date"], keep="last"
    )
    return df.reset_index(drop=True)


def normalize_fina_indicator(fina_df: pd.DataFrame) -> pd.DataFrame:
    if fina_df.empty:
        return pd.DataFrame()
    df = fina_df.copy()
    for col in ["ann_date", "end_date"]:
        if col in df.columns:
            df[col] = parse_date_series(df[col])
    for col in ["eps", "dt_eps", "profit_dedt", "q_dt_roe", "q_npta", "netprofit_yoy", "dt_netprofit_yoy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["actual_np"] = df["profit_dedt"] if "profit_dedt" in df.columns else np.nan
    df["actual_eps"] = df["dt_eps"].fillna(df["eps"]) if "dt_eps" in df.columns else df.get("eps")
    df = df.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(
        subset=["ts_code", "end_date", "ann_date"], keep="last"
    )
    return df.reset_index(drop=True)


def normalize_daily_basic(daily_basic_df: pd.DataFrame) -> pd.DataFrame:
    if daily_basic_df.empty:
        return pd.DataFrame()
    df = daily_basic_df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = parse_date_series(df["trade_date"])
    numeric_cols = [
        "close",
        "turnover_rate",
        "turnover_rate_f",
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "circ_mv",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    if "turnover_rate" in df.columns:
        df["turnover20"] = (
            df.groupby("ts_code")["turnover_rate"]
            .rolling(20, min_periods=15)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        df["turnover20"] = np.nan
    return df
