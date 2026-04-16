from __future__ import annotations

import numpy as np
import pandas as pd


def _quarter_from_date(date_series: pd.Series) -> pd.Series:
    return ((date_series.dt.month - 1) // 3 + 1).astype("Int64")


def construct_earnings_surprise(earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build earnings surprise proxy:
    ES = (Actual - Expected) / |Expected|

    Expected earnings proxy (default for this course project):
    previous-year same-quarter earnings of the same firm.
    """
    if earnings_df.empty:
        return pd.DataFrame()

    df = earnings_df.copy()
    df["actual_earnings"] = pd.to_numeric(df["actual_earnings"], errors="coerce")
    df = df.dropna(subset=["ts_code", "ann_date", "end_date", "actual_earnings"])
    df["fiscal_year"] = df["end_date"].dt.year
    df["fiscal_quarter"] = _quarter_from_date(df["end_date"])
    df = df.sort_values(["ts_code", "fiscal_year", "fiscal_quarter", "ann_date"])

    # Convert cumulative earnings to single-quarter earnings proxy.
    # In many CN financial statement fields, quarterly values are cumulative in-year.
    prev_q = df.groupby("ts_code")["actual_earnings"].shift(1)
    same_year_prev_q = df.groupby("ts_code")["fiscal_year"].shift(1) == df["fiscal_year"]
    df["actual_earnings_single_q"] = np.where(
        same_year_prev_q,
        df["actual_earnings"] - prev_q,
        df["actual_earnings"],
    )
    df["actual_earnings_single_q"] = df["actual_earnings_single_q"].fillna(df["actual_earnings"])

    anchor = df[["ts_code", "fiscal_year", "fiscal_quarter", "actual_earnings"]].copy()
    anchor = anchor.rename(columns={"actual_earnings": "expected_earnings"})
    anchor["fiscal_year"] = anchor["fiscal_year"] + 1

    out = df.merge(anchor, on=["ts_code", "fiscal_year", "fiscal_quarter"], how="left")
    out["earnings_surprise_raw"] = (
        (out["actual_earnings"] - out["expected_earnings"])
        / out["expected_earnings"].abs().replace(0, np.nan)
    )
    # Report-inspired proxy: single-quarter YoY surprise.
    single_q_anchor = (
        out[["ts_code", "fiscal_year", "fiscal_quarter", "actual_earnings_single_q"]]
        .rename(columns={"actual_earnings_single_q": "expected_earnings_single_q"})
        .copy()
    )
    single_q_anchor["fiscal_year"] = single_q_anchor["fiscal_year"] + 1
    out = out.merge(
        single_q_anchor,
        on=["ts_code", "fiscal_year", "fiscal_quarter"],
        how="left",
    )
    out["earnings_surprise_single_q"] = (
        (out["actual_earnings_single_q"] - out["expected_earnings_single_q"])
        / out["expected_earnings_single_q"].abs().replace(0, np.nan)
    )

    out["earnings_surprise"] = out["earnings_surprise_single_q"].fillna(out["earnings_surprise_raw"])

    # Keep usable events and avoid duplicated announcement events.
    out = out.dropna(subset=["earnings_surprise"])
    out = out.sort_values(["ts_code", "ann_date", "end_date"]).drop_duplicates(
        subset=["ts_code", "ann_date"],
        keep="last",
    )
    out = out.reset_index(drop=True)
    return out
