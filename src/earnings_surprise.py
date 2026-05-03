from __future__ import annotations

import numpy as np
import pandas as pd


def _quarter_from_date(date_series: pd.Series) -> pd.Series:
    return ((date_series.dt.month - 1) // 3 + 1).astype("Int64")


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.empty:
        return s
    q_low = s.quantile(lower)
    q_high = s.quantile(upper)
    return s.clip(lower=q_low, upper=q_high)


def construct_earnings_surprise(earnings_df: pd.DataFrame, config: ProjectConfig = None) -> pd.DataFrame:
    """
    Build earnings surprise proxies.
    
    Metrics:
    - surprise_raw: Actual - Expected (in currency units)
    - surprise_pct: (Actual - Expected) / |Expected|
    - surprise_std: Standardized surprise (scaled by cross-sectional SD or winsorized units)
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
    
    # 1. Raw Surprise
    out["surprise_raw"] = out["actual_earnings"] - out["expected_earnings"]
    
    # 2. Percentage Surprise
    out["surprise_pct"] = (
        (out["actual_earnings"] - out["expected_earnings"])
        / out["expected_earnings"].abs().replace(0, np.nan)
    )

    # Standardized proxy for this pipeline (using winsorized pct surprise as baseline)
    w_lower = config.winsor_lower if config else 0.01
    w_upper = config.winsor_upper if config else 0.99
    
    out["surprise_pct_w"] = winsorize_series(out["surprise_pct"], lower=w_lower, upper=w_upper)
    std_val = out["surprise_pct_w"].std()
    out["surprise_std"] = out["surprise_pct_w"] / (std_val if std_val > 0 else 1.0)

    # Legacy compatibility
    out["earnings_surprise"] = out["surprise_pct_w"]

    # Keep usable events
    out = out.dropna(subset=["surprise_pct"])
    out = out.sort_values(["ts_code", "ann_date", "end_date"]).drop_duplicates(
        subset=["ts_code", "ann_date"],
        keep="last",
    )
    return out.reset_index(drop=True)
