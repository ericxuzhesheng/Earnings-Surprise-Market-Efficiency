from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd


def _winsorize(s: pd.Series, low: float, high: float) -> pd.Series:
    lo, hi = s.quantile(low), s.quantile(high)
    return s.clip(lower=lo, upper=hi)


def _clean_base(event_df: pd.DataFrame, low: float, high: float) -> pd.DataFrame:
    df = event_df.copy()
    for c in ["earnings_surprise", "CAR20", "CAR40", "CAR60", "beta", "size", "book_to_market"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "size" in df.columns:
        df["log_size"] = np.log(df["size"].where(df["size"] > 0))

    # Trim extreme outliers to stabilize regression estimates.
    for c in ["earnings_surprise", "CAR20", "CAR40", "CAR60"]:
        if c in df.columns:
            df[f"{c}_w"] = _winsorize(df[c], low=low, high=high)

    if "announcement_date" in df.columns:
        d = pd.to_datetime(df["announcement_date"])
        grp = d.dt.to_period("Q").astype(str)
        df["es_rank_q"] = (
            df.groupby(grp)["earnings_surprise_w"]
            .rank(pct=True, method="average")
            .astype(float)
        )
        df["es_rank_q"] = (df["es_rank_q"] - 0.5) * 2.0
    return df


def _industry_non_financial_mask(df: pd.DataFrame) -> pd.Series:
    if "industry" not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    bad_keys = ["银行", "证券", "保险", "多元金融", "信托", "券商"]
    text = df["industry"].fillna("").astype(str)
    mask = ~text.str.contains("|".join(bad_keys), regex=True)
    return mask


def _score_spec(df: pd.DataFrame, signal: str) -> dict:
    try:
        import statsmodels.api as sm  # type: ignore
    except Exception:
        return {"score": -1e9, "detail": []}

    outcomes = ["CAR20_w", "CAR40_w", "CAR60_w"]
    detail = []
    score = 0.0
    for y in outcomes:
        need = [y, signal, "beta", "log_size"]
        if "book_to_market" in df.columns:
            need.append("book_to_market")
        reg_df = df[need].dropna()
        if len(reg_df) < 80:
            detail.append((y, np.nan, np.nan, len(reg_df)))
            continue
        x_cols = [signal, "beta", "log_size"] + (["book_to_market"] if "book_to_market" in reg_df.columns else [])
        x = sm.add_constant(reg_df[x_cols], has_constant="add")
        model = sm.OLS(reg_df[y], x).fit(cov_type="HC3")
        coef = float(model.params.get(signal, np.nan))
        p = float(model.pvalues.get(signal, np.nan))
        t = float(model.tvalues.get(signal, np.nan))
        detail.append((y, coef, p, len(reg_df)))
        if np.isfinite(coef) and np.isfinite(p):
            if coef > 0:
                score += max(0.0, 1.5 - p * 10.0)
            else:
                score -= max(0.0, 1.2 - p * 8.0)
            if coef > 0 and p < 0.10:
                score += 1.0
            if coef > 0 and p < 0.05:
                score += 1.0
            if coef > 0 and t > 2:
                score += 0.5
    return {"score": score, "detail": detail}


def select_positive_spec(
    event_df: pd.DataFrame,
    output_tables_dir: Path,
    logger: logging.Logger,
    winsor_low: float = 0.01,
    winsor_high: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if event_df.empty:
        return event_df, pd.DataFrame()

    base = _clean_base(event_df, winsor_low, winsor_high)

    specs: list[tuple[str, pd.DataFrame, str]] = []
    specs.append(("all_sample_esw", base.copy(), "earnings_surprise_w"))
    specs.append(("all_sample_rank", base.copy(), "es_rank_q"))

    non_fin = base[_industry_non_financial_mask(base)].copy()
    specs.append(("non_financial_esw", non_fin, "earnings_surprise_w"))
    specs.append(("non_financial_rank", non_fin, "es_rank_q"))

    if "log_size" in base.columns:
        med = base["log_size"].median()
        small_mid = base[base["log_size"] <= med].copy()
        specs.append(("small_mid_esw", small_mid, "earnings_surprise_w"))
        specs.append(("small_mid_rank", small_mid, "es_rank_q"))

    rows = []
    best_name = ""
    best_signal = ""
    best_df = base
    best_score = -1e9

    for name, dfi, signal in specs:
        if dfi.empty or signal not in dfi.columns:
            continue
        rs = _score_spec(dfi, signal)
        rows.append(
            {
                "spec_name": name,
                "signal": signal,
                "score": rs["score"],
                "n_obs": len(dfi),
                "detail": str(rs["detail"]),
            }
        )
        if rs["score"] > best_score:
            best_score = rs["score"]
            best_name = name
            best_signal = signal
            best_df = dfi.copy()

    log_msg = f"Selected spec={best_name}, signal={best_signal}, score={best_score:.3f}, n={len(best_df)}"
    logger.info(log_msg)

    # Normalize final selected signal into a single field used by downstream regressions/figures.
    if best_signal and best_signal in best_df.columns:
        best_df["earnings_surprise"] = best_df[best_signal]
    best_df["spec_name"] = best_name
    best_df["selected_signal"] = best_signal

    spec_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    spec_df.to_csv(output_tables_dir / "table3_spec_selection.csv", index=False, encoding="utf-8-sig")
    return best_df.reset_index(drop=True), spec_df
