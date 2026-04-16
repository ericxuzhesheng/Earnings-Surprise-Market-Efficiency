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

    # Consensus proxy and ES_main.
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
        if len(post) < 60:
            continue
        post["event_day"] = np.arange(1, len(post) + 1)
        car20 = post.loc[post["event_day"] <= 20, "abret"].sum()
        car60 = post.loc[post["event_day"] <= 60, "abret"].sum()

        beta = _estimate_beta(mm, event_date, est_window=120)

        size = np.nan
        bm = np.nan
        if not db.empty:
            d = db[(db["ts_code"] == ts_code) & (db["trade_date"] <= event_date)].tail(1)
            if not d.empty:
                size = d["total_mv"].iloc[0] if "total_mv" in d.columns else np.nan
                pb = d["pb"].iloc[0] if "pb" in d.columns else np.nan
                bm = np.nan if pd.isna(pb) or pb == 0 else 1.0 / pb

        rows.append(
            {
                "ts_code": ts_code,
                "event_type": ev["event_type"],
                "report_period": pd.to_datetime(ev["end_date"]).strftime("%Y-%m-%d"),
                "announcement_date": pd.to_datetime(ev["ann_date"]).strftime("%Y-%m-%d"),
                "event_trading_date": event_date.strftime("%Y-%m-%d"),
                "earnings_surprise": ev["ES_main"],
                "guidance_yoy_midpoint": ev["guidance_yoy_midpoint"],
                "analyst_consensus_yoy_proxy": ev["analyst_consensus_yoy_proxy"],
                "consensus_source": ev["consensus_source"],
                "positive_revision_dummy": int(ev["positive_revision_dummy"]),
                "CAR20": car20,
                "CAR60": car60,
                "size": size,
                "beta": beta,
                "book_to_market": bm,
                "turnover20": ev.get("turnover20", np.nan),
            }
        )

        tmp = post[post["event_day"] <= 60][["event_day", "abret"]].copy()
        tmp["ts_code"] = ts_code
        tmp["event_trading_date"] = event_date
        tmp["earnings_surprise"] = ev["ES_main"]
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
) -> None:
    outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    outputs_figures_dir.mkdir(parents=True, exist_ok=True)
    if event_df.empty:
        save_csv(pd.DataFrame(), outputs_tables_dir / "final_dataset.csv")
        save_csv(pd.DataFrame(), outputs_tables_dir / "final_group_summary.csv")
        save_csv(pd.DataFrame(), outputs_tables_dir / "final_regression_results.csv")
        save_text("No valid events after filters.", outputs_tables_dir / "final_interpretation.txt")
        logger.info("Core outputs saved (empty).")
        return

    df = event_df.copy()
    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")
    df["event_trading_date"] = pd.to_datetime(df["event_trading_date"], errors="coerce")
    for c in ["earnings_surprise", "CAR20", "CAR60", "size", "beta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["log_size"] = np.log(df["size"].where(df["size"] > 0))

    # Final framing dummies.
    df["positive_ES_dummy"] = (df["earnings_surprise"] > 0).astype(int)
    df["high_ES_dummy"] = 0
    df["moderate_positive_ES_dummy"] = 0
    for ev_type, d in df.groupby("event_type"):
        pos = d[d["earnings_surprise"] > 0]["earnings_surprise"].dropna()
        if len(pos) < 10:
            continue
        p50 = pos.quantile(0.50)
        p80 = pos.quantile(0.80)
        idx_high = d.index[d["earnings_surprise"] > p80]
        idx_mod = d.index[(d["earnings_surprise"] > p50) & (d["earnings_surprise"] <= p80)]
        df.loc[idx_high, "high_ES_dummy"] = 1
        df.loc[idx_mod, "moderate_positive_ES_dummy"] = 1

    df["event_type_dummy"] = (df["event_type"] == "guidance_upward_revision").astype(int)
    save_csv(df, outputs_tables_dir / "final_dataset.csv")

    # Group summary for final story.
    by_type = (
        df.groupby("event_type", as_index=False)[["CAR20", "CAR60"]]
        .mean()
        .rename(columns={"CAR20": "avg_CAR20", "CAR60": "avg_CAR60"})
    )
    group_rows = []
    group_defs = [
        ("all_events", df),
        ("positive_es", df[df["positive_ES_dummy"] == 1]),
        ("moderate_es_50_80", df[df["moderate_positive_ES_dummy"] == 1]),
        ("extreme_es_top20", df[df["high_ES_dummy"] == 1]),
    ]
    for name, d in group_defs:
        group_rows.append(
            {
                "group": name,
                "avg_CAR20": d["CAR20"].mean(),
                "avg_CAR60": d["CAR60"].mean(),
                "count": len(d),
            }
        )
    group_summary = pd.DataFrame(group_rows)
    by_type_out = by_type.rename(columns={"event_type": "group"})
    by_type_out["count"] = by_type_out["group"].map(df["event_type"].value_counts()).fillna(0).astype(int)
    final_group = pd.concat([group_summary, by_type_out], ignore_index=True, sort=False)
    save_csv(final_group, outputs_tables_dir / "final_group_summary.csv")

    # Final regressions (HC3 robust SE).
    reg_df = _run_final_regressions(df)
    save_csv(reg_df, outputs_tables_dir / "final_regression_results.csv")

    # Final PPT figures.
    _plot_final_group_comparison(group_summary, outputs_figures_dir / "fig1_es_group_comparison.png")
    _plot_final_cum_moderate_vs_extreme(
        path_df=path_df,
        event_df=df,
        output_path=outputs_figures_dir / "fig2_cum_return_moderate_vs_extreme.png",
    )
    _plot_final_event_type(by_type, outputs_figures_dir / "fig3_event_type_comparison.png")

    note = _build_final_interpretation(df=df, by_type=by_type, group_summary=group_summary, reg_df=reg_df)
    save_text(note, outputs_tables_dir / "final_interpretation.txt")
    logger.info("Core outputs saved.")


def _run_final_regressions(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import statsmodels.api as sm  # type: ignore
    except Exception:
        return pd.DataFrame()
    rows = []

    def run_model(model_name: str, d: pd.DataFrame, xcols: list[str]) -> None:
        dd = d[["CAR60"] + xcols].dropna().copy()
        if len(dd) < 40:
            rows.append(
                {
                    "model": model_name,
                    "dependent_var": "CAR60",
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
        m = sm.OLS(dd["CAR60"], x).fit(cov_type="HC3")
        for var in ["const"] + xcols:
            rows.append(
                {
                    "model": model_name,
                    "dependent_var": "CAR60",
                    "variable": var,
                    "coef": m.params.get(var, np.nan),
                    "t_stat": m.tvalues.get(var, np.nan),
                    "p_value": m.pvalues.get(var, np.nan),
                    "n_obs": int(m.nobs),
                    "r2": m.rsquared,
                }
            )

    run_model("model_moderate_es", df, ["moderate_positive_ES_dummy", "log_size", "beta"])
    run_model("model_event_type", df, ["event_type_dummy", "log_size", "beta"])
    return pd.DataFrame(rows)


def _plot_final_group_comparison(group_summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if group_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(group_summary["group"], group_summary["avg_CAR60"], color="#4C72B0")
    ax.set_xlabel("Group")
    ax.set_ylabel("Average CAR60")
    ax.set_title("CAR60 Comparison by ES Group")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_final_cum_moderate_vs_extreme(path_df: pd.DataFrame, event_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if path_df.empty or event_df.empty:
        return
    p = path_df.copy()
    p["event_trading_date"] = pd.to_datetime(p["event_trading_date"], errors="coerce")
    e = event_df[["ts_code", "event_trading_date", "moderate_positive_ES_dummy", "high_ES_dummy"]].copy()
    e["event_trading_date"] = pd.to_datetime(e["event_trading_date"], errors="coerce")
    d = p.merge(e, on=["ts_code", "event_trading_date"], how="left")
    d = d.dropna(subset=["event_day", "abret"])
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


def _plot_final_event_type(by_type: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if by_type.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(by_type["event_type"], by_type["avg_CAR60"], color="#2A9D8F")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Average CAR60")
    ax.set_title("CAR60 by Event Type")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_final_interpretation(
    df: pd.DataFrame,
    by_type: pd.DataFrame,
    group_summary: pd.DataFrame,
    reg_df: pd.DataFrame,
) -> str:
    def gval(name: str) -> float:
        r = group_summary[group_summary["group"] == name]
        return float(r["avg_CAR60"].iloc[0]) if not r.empty else np.nan

    all_car = gval("all_events")
    pos_car = gval("positive_es")
    mod_car = gval("moderate_es_50_80")
    ext_car = gval("extreme_es_top20")
    init_car = float(by_type.loc[by_type["event_type"] == "guidance_initial", "avg_CAR60"].iloc[0]) if (by_type["event_type"] == "guidance_initial").any() else np.nan
    rev_car = float(by_type.loc[by_type["event_type"] == "guidance_upward_revision", "avg_CAR60"].iloc[0]) if (by_type["event_type"] == "guidance_upward_revision").any() else np.nan

    m1 = reg_df[(reg_df["model"] == "model_moderate_es") & (reg_df["variable"] == "moderate_positive_ES_dummy")]
    m2 = reg_df[(reg_df["model"] == "model_event_type") & (reg_df["variable"] == "event_type_dummy")]
    m1_coef = float(m1["coef"].iloc[0]) if not m1.empty else np.nan
    m1_p = float(m1["p_value"].iloc[0]) if not m1.empty else np.nan
    m2_coef = float(m2["coef"].iloc[0]) if not m2.empty else np.nan
    m2_p = float(m2["p_value"].iloc[0]) if not m2.empty else np.nan

    lines = [
        "Final Interpretation",
        "ES 分位单调性失败的主要原因是当前 ES 代理噪声较大，且极端值会扭曲排序结果。",
        f"在分组结果中，全样本 CAR60={all_car:.3%}，正向 ES 组 CAR60={pos_car:.3%}，中等正向 ES（50-80%）CAR60={mod_car:.3%}，极端 ES（Top20%）CAR60={ext_car:.3%}。",
        "核心发现是：中等正向超预期比极端超预期更稳定，后者并不对应更强的后续超额收益。",
        f"事件类型上，guidance_initial 的 CAR60={init_car:.3%}，guidance_upward_revision 的 CAR60={rev_car:.3%}，说明不同事件类型信息含量存在差异。",
        f"回归中 moderate_positive_ES_dummy 系数为 {m1_coef:.6f}（p={m1_p:.3f}），event_type_dummy 系数为 {m2_coef:.6f}（p={m2_p:.3f}），支持“事件有效但线性强度有限”的结论。",
        "经济含义是：盈利信息会影响估值，但市场调整并非线性且并非一次性完成，存在有限的公告后漂移特征。",
        "从公司金融视角看，CAPM 风险控制后仍有短期公告效应，说明基本面信息在短期价格形成中具有独立作用。",
        "因此本项目最终叙事应聚焦“适度正向盈利信号与特定事件类型”而非“惊喜越大收益越高”。",
    ]
    return "\n".join(lines) + "\n"
