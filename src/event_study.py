from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.config import ProjectConfig


def _compute_beta(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    event_date: pd.Timestamp,
    estimation_window: int,
    buffer_days: int,
) -> float:
    merged = stock_df[["trade_date", "ret"]].merge(
        market_df[["trade_date", "mkt_ret"]], on="trade_date", how="inner"
    )
    merged = merged.sort_values("trade_date").dropna(subset=["ret", "mkt_ret"])

    end_date = event_date - pd.Timedelta(days=buffer_days)
    hist = merged[merged["trade_date"] < end_date].tail(estimation_window)
    if len(hist) < max(30, estimation_window // 3):
        return np.nan

    x = hist["mkt_ret"].to_numpy(dtype=float)
    y = hist["ret"].to_numpy(dtype=float)
    var_x = np.var(x)
    if var_x == 0:
        return np.nan
    beta = np.cov(x, y, ddof=0)[0, 1] / var_x
    return float(beta)


def _compute_industry_returns(
    prices_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
) -> pd.DataFrame:
    """Equal-weighted mean daily return per industry per trading date."""
    if stocks_df.empty or "industry" not in stocks_df.columns:
        return pd.DataFrame()

    industry_map = stocks_df.set_index("ts_code")["industry"].dropna().to_dict()
    prices = prices_df[["ts_code", "trade_date", "ret"]].copy()
    prices["industry"] = prices["ts_code"].map(industry_map)
    prices = prices.dropna(subset=["industry", "ret"])
    if prices.empty:
        return pd.DataFrame()

    return (
        prices.groupby(["trade_date", "industry"])["ret"]
        .mean()
        .reset_index()
        .rename(columns={"ret": "industry_ret"})
    )


def _get_event_window_abnormal_returns(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    event_date: pd.Timestamp,
    pre_window: int = 10,
    post_window: int = 60,
) -> pd.DataFrame:
    merged = stock_df[["trade_date", "ret"]].merge(
        market_df[["trade_date", "mkt_ret"]], on="trade_date", how="inner"
    )
    merged = merged.sort_values("trade_date").dropna(subset=["ret", "mkt_ret"])
    merged["abret"] = merged["ret"] - merged["mkt_ret"]

    # Identify the event trading day (the day of announcement or the next available trading day)
    event_trading_days = merged[merged["trade_date"] >= event_date]
    if event_trading_days.empty:
        return pd.DataFrame()
    
    event_idx = event_trading_days.index[0]
    
    # Get relative position
    # Find numeric index in the full sorted merged dataframe
    merged_reset = merged.reset_index(drop=True)
    numeric_event_idx = merged_reset[merged_reset["trade_date"] >= event_date].index[0]
    
    start_idx = max(0, numeric_event_idx - pre_window)
    end_idx = min(len(merged_reset), numeric_event_idx + post_window + 1)
    
    window_df = merged_reset.iloc[start_idx:end_idx].copy()
    window_df["event_day"] = np.arange(start_idx, end_idx) - numeric_event_idx
    
    return window_df


def build_event_level_dataset(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    daily_basic_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    config: ProjectConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events_df.empty or prices_df.empty or market_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    event_rows = []
    path_rows = []
    market = market_df.copy()
    market = market.dropna(subset=["trade_date", "mkt_ret"]).sort_values("trade_date")

    # Pre-compute industry-level daily returns for the industry-adjusted model.
    industry_returns_df = _compute_industry_returns(prices_df, stocks_df)
    has_industry_model = not industry_returns_df.empty

    for _, event in events_df.iterrows():
        ts_code = event["ts_code"]
        ann_date = pd.to_datetime(event["ann_date"])
        stock_df = prices_df[prices_df["ts_code"] == ts_code].copy()
        if stock_df.empty:
            continue
        stock_df = stock_df.dropna(subset=["trade_date", "ret"]).sort_values("trade_date")

        post = _get_event_window_abnormal_returns(stock_df, market, ann_date, pre_window=10, post_window=60)
        if post.empty:
            continue

        beta = _compute_beta(
            stock_df=stock_df,
            market_df=market,
            event_date=ann_date,
            estimation_window=config.beta_estimation_window,
            buffer_days=config.beta_buffer_days,
        )

        stock_meta = stocks_df.loc[stocks_df["ts_code"] == ts_code].head(1)
        stock_name = stock_meta["name"].iloc[0] if not stock_meta.empty else np.nan
        industry = stock_meta["industry"].iloc[0] if not stock_meta.empty else np.nan

        car_values = {}
        # 1. Legacy windows (CAR1..CAR60) - mapped to CAR[1, w]
        for w in config.event_windows:
            car_values[f"CAR{w}"] = post.loc[(post["event_day"] >= 1) & (post["event_day"] <= w), "abret"].sum()

        # 2. New diagnostic windows
        for start, end in config.leakage_windows:
            car_values[f"CAR_{start}_{end}"] = post.loc[(post["event_day"] >= start) & (post["event_day"] <= end), "abret"].sum()
        for start, end in config.immediate_windows:
            car_values[f"CAR_{start}_{end}"] = post.loc[(post["event_day"] >= start) & (post["event_day"] <= end), "abret"].sum()
        for start, end in config.drift_windows:
            car_values[f"CAR_{start}_{end}"] = post.loc[(post["event_day"] >= start) & (post["event_day"] <= end), "abret"].sum()

        # 3. Industry-adjusted return model: AR = R_it - R_industry,t (equal-weighted).
        if has_industry_model and pd.notna(industry) and industry in industry_returns_df["industry"].values:
            ind_ret_series = (
                industry_returns_df[industry_returns_df["industry"] == industry]
                .set_index("trade_date")["industry_ret"]
            )
            post_ind = post.copy()
            post_ind["ia_abret"] = post_ind["ret"] - post_ind["trade_date"].map(ind_ret_series).fillna(post_ind["mkt_ret"])
            for w in config.event_windows:
                car_values[f"IAR{w}"] = post_ind.loc[
                    (post_ind["event_day"] >= 1) & (post_ind["event_day"] <= w), "ia_abret"
                ].sum()
            for start, end in (*config.immediate_windows, *config.drift_windows):
                car_values[f"IAR_{start}_{end}"] = post_ind.loc[
                    (post_ind["event_day"] >= start) & (post_ind["event_day"] <= end), "ia_abret"
                ].sum()

        # Event-date characteristic proxy: nearest prior trading day from daily_basic.
        size = np.nan
        bm = np.nan
        if not daily_basic_df.empty:
            char_df = daily_basic_df[daily_basic_df["ts_code"] == ts_code].copy()
            char_df = char_df[char_df["trade_date"] <= ann_date].sort_values("trade_date")
            if not char_df.empty:
                last_row = char_df.iloc[-1]
                size = last_row.get("total_mv", np.nan)
                pb = last_row.get("pb", np.nan)
                bm = np.nan if pd.isna(pb) or pb == 0 else 1.0 / pb

        row = {
            "ts_code": ts_code,
            "stock_name": stock_name,
            "industry": industry,
            "announcement_date": ann_date,
            "fiscal_year": event.get("fiscal_year", np.nan),
            "fiscal_quarter": event.get("fiscal_quarter", np.nan),
            "actual_earnings": event.get("actual_earnings", np.nan),
            "expected_earnings": event.get("expected_earnings", np.nan),
            "earnings_surprise": event.get("earnings_surprise", np.nan),
            "beta": beta,
            "size": size,
            "book_to_market": bm,
            **car_values,
        }
        event_rows.append(row)

        post_path = post[post["event_day"] <= max(config.event_windows)].copy()
        post_path["ts_code"] = ts_code
        post_path["announcement_date"] = ann_date
        post_path["earnings_surprise"] = event.get("earnings_surprise", np.nan)
        path_rows.append(
            post_path[
                ["ts_code", "announcement_date", "event_day", "abret", "earnings_surprise"]
            ]
        )

    event_df = pd.DataFrame(event_rows)
    if event_df.empty:
        logger.warning("Event-level dataset is empty after event-study filtering.")
        return event_df, pd.DataFrame()

    required_cols = [c for c in ["earnings_surprise", "CAR5", "CAR20", "CAR60"] if c in event_df.columns]
    event_df = event_df.dropna(subset=required_cols)
    event_df["year"] = pd.to_datetime(event_df["announcement_date"]).dt.year
    event_df["quarter"] = pd.to_datetime(event_df["announcement_date"]).dt.quarter
    event_df = event_df.sort_values(["announcement_date", "ts_code"]).reset_index(drop=True)
    logger.info("Final event-level rows: %s", len(event_df))

    path_df = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    return event_df, path_df
