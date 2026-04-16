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


def _get_event_window_abnormal_returns(
    stock_df: pd.DataFrame,
    market_df: pd.DataFrame,
    event_date: pd.Timestamp,
) -> pd.DataFrame:
    merged = stock_df[["trade_date", "ret"]].merge(
        market_df[["trade_date", "mkt_ret"]], on="trade_date", how="inner"
    )
    merged = merged.sort_values("trade_date").dropna(subset=["ret", "mkt_ret"])
    merged["abret"] = merged["ret"] - merged["mkt_ret"]

    # Post-announcement window starts at the next trading day after ann_date.
    post = merged[merged["trade_date"] > event_date].copy().reset_index(drop=True)
    if post.empty:
        return post
    post["event_day"] = np.arange(1, len(post) + 1)
    return post


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

    for _, event in events_df.iterrows():
        ts_code = event["ts_code"]
        ann_date = pd.to_datetime(event["ann_date"])
        stock_df = prices_df[prices_df["ts_code"] == ts_code].copy()
        if stock_df.empty:
            continue
        stock_df = stock_df.dropna(subset=["trade_date", "ret"]).sort_values("trade_date")

        post = _get_event_window_abnormal_returns(stock_df, market, ann_date)
        if post.empty or post["event_day"].max() < max(config.event_windows):
            continue

        beta = _compute_beta(
            stock_df=stock_df,
            market_df=market,
            event_date=ann_date,
            estimation_window=config.beta_estimation_window,
            buffer_days=config.beta_buffer_days,
        )

        car_values = {}
        for w in config.event_windows:
            car_values[f"CAR{w}"] = post.loc[post["event_day"] <= w, "abret"].sum()

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

        stock_meta = stocks_df.loc[stocks_df["ts_code"] == ts_code].head(1)
        stock_name = stock_meta["name"].iloc[0] if not stock_meta.empty else np.nan
        industry = stock_meta["industry"].iloc[0] if not stock_meta.empty else np.nan

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

    event_df = event_df.dropna(subset=["earnings_surprise", "CAR20", "CAR40", "CAR60"])
    event_df["year"] = pd.to_datetime(event_df["announcement_date"]).dt.year
    event_df["quarter"] = pd.to_datetime(event_df["announcement_date"]).dt.quarter
    event_df = event_df.sort_values(["announcement_date", "ts_code"]).reset_index(drop=True)
    logger.info("Final event-level rows: %s", len(event_df))

    path_df = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    return event_df, path_df
