"""Frozen-cache audit. Optional bounded adjustment-factor refresh; no raw overwrite."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.event_validation import calendar_returns, clustered_estimate, has_contamination, match_consensus, window_car
from src.tushare_normalization import normalize_forecast, normalize_report_rc, parse_date_series


def run(refresh_factors=False, matching_only=False, expanded=False):
    cache = ROOT / ("data_raw/expansion_20260930" if expanded else "data_raw/cache")
    output = ROOT / ("outputs/expanded_20260930" if expanded else "outputs/event_validation")
    output.mkdir(parents=True, exist_ok=True)
    inputs = []
    if expanded:
        expansion_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if expansion_manifest.get("failures"):
            raise ValueError("Expanded announcement collection has failed partitions; resume collection first")
        if not matching_only:
            market_receipt = json.loads((cache / "market_sources_receipt.json").read_text(encoding="utf-8"))
            if not market_receipt.get("complete"):
                raise ValueError("Expanded market snapshots are incomplete; resume collection first")

    def read(path, endpoint):
        if not path.exists():
            raise FileNotFoundError(f"Required frozen source absent: {path}")
        data = pd.read_csv(path)
        inputs.append({"path": str(path.relative_to(ROOT)), "endpoint": endpoint, "rows": len(data),
                       "possible_page_cap": endpoint == "report_rc" and len(data) >= 5000})
        return data

    universe = read(cache / "stock_basic/listed.csv", "stock_basic").sort_values("ts_code")
    if not expanded:
        universe = universe.head(300)
    codes = set(universe.ts_code)
    market = read(cache / ("market/market_with_warmup.csv" if expanded else "market/market_399300_SZ_20200101_20260421.csv"), "market")
    market["trade_date"] = parse_date_series(market.trade_date)
    market = market.sort_values("trade_date").drop_duplicates("trade_date")
    if expanded:
        exchange_calendar = read(cache / "market/calendar_with_warmup.csv", "trade_cal")
        open_dates = parse_date_series(exchange_calendar.loc[exchange_calendar.is_open.eq(1), "cal_date"]).sort_values()
        if not np.array_equal(market.trade_date.to_numpy(), open_dates.to_numpy()):
            raise ValueError("Market benchmark does not cover the exchange trading calendar")
    market["mkt_ret"] = market.close.pct_change(fill_method=None)
    calendar = pd.DatetimeIndex(market.trade_date)
    start, end = calendar.min(), calendar.max()
    event_start = pd.Timestamp(expansion_manifest['start']) if expanded else start
    forecasts = []
    fiscal_periods = pd.period_range(event_start - pd.offsets.QuarterEnd(), end + pd.offsets.QuarterEnd(), freq="Q") if expanded else pd.period_range(start, end, freq="Q")
    for period in fiscal_periods:
        path = cache / "forecast_vip" / f"{period.end_time:%Y%m%d}.csv"
        if path.exists():
            frame = read(path, "forecast" if expanded else "forecast_vip")
            forecasts.append(frame[frame.ts_code.isin(codes)])
    raw_forecasts = pd.concat(forecasts, ignore_index=True)
    if expanded:
        raw_forecasts = raw_forecasts.loc[parse_date_series(raw_forecasts.ann_date) <= end]
    forecast_rows = len(raw_forecasts)
    # Duplicate ingestion cannot create a spurious revision. Conflicting same-date
    # financial values have no reliable ordering in the source and are excluded.
    key = ["ts_code", "end_date", "ann_date"]
    value = ["net_profit_min", "net_profit_max"]
    conflict = raw_forecasts.groupby(key)[value].transform("nunique").gt(1).any(axis=1)
    conflict_count = int(conflict.sum())
    ambiguous_periods = pd.MultiIndex.from_frame(raw_forecasts.loc[conflict, ["ts_code", "end_date"]])
    ambiguous_period = pd.MultiIndex.from_frame(raw_forecasts[["ts_code", "end_date"]]).isin(ambiguous_periods)
    unique = raw_forecasts.loc[~ambiguous_period].drop_duplicates(key)
    events = normalize_forecast(unique)
    all_forecast_events = events.copy()
    events = events.loc[~events.is_revision & events.ann_date.between(event_start, end)].copy()
    events["period_end"] = events.end_date
    positions = calendar.searchsorted(events.ann_date, side="right")
    events = events.loc[positions < len(calendar)].copy()
    events["event_trade_date"] = calendar[positions[positions < len(calendar)]].to_numpy()
    events = events.sort_values(["ts_code", "ann_date", "end_date"]).reset_index(drop=True)
    events["event_id"] = np.arange(len(events)) + 1

    reports = []
    complete_reports = (output / "source_report_rc/receipt.json").exists()
    if complete_reports:
        for code in sorted(codes):
            frame = read(output / "source_report_rc" / f"{code}.csv", "report_rc_complete")
            if not frame.empty:
                reports.append(frame)
    else:
        for month in pd.period_range(start, end, freq="M"):
            report_cache = ROOT / "data_raw/cache" if expanded else cache
            path = report_cache / "report_rc" / f"{month.start_time:%Y%m%d}_{month.end_time:%Y%m%d}.csv"
            if expanded and (cache / "report_rc_complete" / f"{month.strftime('%Y%m')}.csv").exists():
                continue
            if path.exists():
                frame = read(path, "report_rc")
                reports.append(frame.loc[frame.ts_code.isin(codes)])
        if expanded:
            for path in sorted((cache / "report_rc_complete").glob('*.csv')):
                frame = read(path, "report_rc_added_complete_partition")
                reports.append(frame.loc[frame.ts_code.isin(codes)])
            for path in sorted((cache / "report_rc_pages").glob('*/*.csv')):
                if (cache / 'report_rc_complete' / f'{path.parent.name}.csv').exists():
                    continue
                frame = read(path, "report_rc_added_partial_page")
                reports.append(frame.loc[frame.ts_code.isin(codes)])
    reports = normalize_report_rc(pd.concat(reports, ignore_index=True).drop_duplicates())
    matched = events.merge(match_consensus(events, reports), on="event_id", validate="one_to_one")
    if matching_only:
        matched.to_csv(output / "matching_audit.csv", index=False)
        print(f"Prepared {matched.surprise_pct.notna().sum()} matched events for bounded source repair")
        return
    baseline = match_consensus(events, reports, before_announcement=False).add_prefix("legacy_")
    matched = matched.merge(baseline, left_on="event_id", right_on="legacy_event_id", validate="one_to_one")
    strict180 = match_consensus(events, reports, freshness_days=180).set_index("event_id")
    matched["surprise_180"] = matched.event_id.map(strict180.surprise_pct)
    matched["brokers_180"] = matched.event_id.map(strict180.matched_brokers)
    matched["legacy_news_contaminated"] = matched.legacy_latest_report_date.ge(matched.ann_date)
    matched["consensus_changed"] = (matched.expected_np - matched.legacy_expected_np).abs().gt(1e-8)
    matched["period_label"] = matched.period_end.dt.strftime("%m-%d")
    matched_codes = sorted(matched.loc[matched.surprise_pct.notna(), "ts_code"].unique())
    factor_dir = cache / "source_adj_factor" if expanded else output / "source_factors"
    factor_dir.mkdir(exist_ok=True)
    if refresh_factors:
        import tushare as ts
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            raise RuntimeError("TUSHARE_TOKEN required only for --refresh-factors")

        def refresh(code):
            path = factor_dir / f"{code}.csv"
            if path.exists():
                return
            # Each request is bounded to one stock and the frozen price span.
            data = ts.pro_api(token).adj_factor(ts_code=code, start_date=f"{start:%Y%m%d}", end_date=f"{end:%Y%m%d}")
            if data.empty:
                raise ValueError(f"No adjustment factors for {code}")
            data.to_csv(path, index=False)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(refresh, matched_codes))
        (factor_dir / "receipt.json").write_text(json.dumps({"retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": "adj_factor", "codes": matched_codes, "start_date": str(start.date()), "end_date": str(end.date())}, indent=2), encoding="utf-8")
        print(f"Adjustment-factor snapshots: {len(matched_codes)} stocks", flush=True)

    # Keep other earnings release types as contamination flags, never pool
    # incompatible deducted-profit or EPS definitions into the main signal.
    other = [all_forecast_events[["ts_code", "ann_date"]]]
    for period in fiscal_periods:
        path = cache / "express_vip" / f"{period.end_time:%Y%m%d}.csv"
        if path.exists():
            frame = read(path, "express" if expanded else "express_vip")
            other.append(frame.loc[frame.ts_code.isin(matched_codes), ["ts_code", "ann_date"]])
    if expanded:
        financial_path = cache / "financial_announcements.parquet"
        financial = pd.read_parquet(financial_path)
        inputs.append({"path":str(financial_path.relative_to(ROOT)),"endpoint":"income",
                       "rows":len(financial),"possible_page_cap":False})
        other.append(financial[["ts_code","ann_date"]])
        other.append(financial[["ts_code","f_ann_date"]].rename(columns={"f_ann_date":"ann_date"}))
    for code in ([] if expanded else matched_codes):
        refreshed = output / "source_fina_indicator" / f"{code}.csv"
        path = refreshed if refreshed.exists() else cache / "fina_indicator" / f"{code.replace('.', '_')}.csv"
        if path.exists():
            frame = read(path, "fina_indicator")
            if not frame.empty:
                other.append(frame[["ts_code", "ann_date"]])
    announcements = pd.concat(other, ignore_index=True)
    announcements["ann_date"] = parse_date_series(announcements.ann_date)
    apos = calendar.searchsorted(announcements.ann_date, side="right")
    announcements = announcements.loc[apos < len(calendar)].copy()
    announcements["event_trade_date"] = calendar[apos[apos < len(calendar)]].to_numpy()
    dates = announcements.groupby("ts_code").event_trade_date.agg(lambda s: sorted(set(s)))
    same_day_periods = events.groupby(["ts_code", "event_trade_date"]).period_end.nunique()
    rows = []
    for code in matched_codes:
        fresh_price = (cache if expanded else output) / "source_daily" / f"{code}.csv"
        price = read(fresh_price if fresh_price.exists() else cache / "prices" / f"{code.replace('.', '_')}.csv", "prices")
        price["trade_date"] = parse_date_series(price.trade_date)
        fresh_basic = (cache if expanded else output) / "source_daily_basic" / f"{code}.csv"
        db = read(fresh_basic if fresh_basic.exists() else cache / "daily_basic" / f"{code.replace('.', '_')}.csv", "daily_basic")
        db["trade_date"] = parse_date_series(db.trade_date)
        db = db.drop_duplicates("trade_date").set_index("trade_date").reindex(calendar)
        turnover = db.get("turnover_rate", pd.Series(np.nan, index=calendar)).rolling(20, min_periods=15).mean().shift(1)
        unadjusted = calendar_returns(price, market)
        factor_path = factor_dir / f"{code}.csv"
        factors = read(factor_path, "adj_factor") if factor_path.exists() else None
        if factors is not None:
            factors["trade_date"] = parse_date_series(factors.trade_date)
            adjusted = calendar_returns(price, market, factors)
        listed = parse_date_series(universe.loc[universe.ts_code.eq(code), "list_date"]).iloc[0]
        delisted = parse_date_series(universe.loc[universe.ts_code.eq(code), "delist_date"]).iloc[0] if "delist_date" in universe else pd.NaT
        for event in matched.loc[matched.ts_code.eq(code) & matched.surprise_pct.notna()].to_dict("records"):
            date = event["event_trade_date"]
            event["turnover20_prior"] = turnover.loc[date]
            event["liquidity_source_available"] = "turnover_rate" in db.columns
            # An old listing before the calendar starts already has sufficient
            # history only when at least 120 calendar sessions are observable.
            age = calendar.searchsorted(date) - calendar.searchsorted(listed)
            old_listing = listed < start - pd.Timedelta(days=365)
            event["listed_at_event"] = bool(listed <= date and (pd.isna(delisted) or date <= delisted))
            event["eligible"] = bool(event["listed_at_event"] and (age >= 120 or old_listing) and turnover.loc[date] >= 0.3)
            event["has_adjustment_factors"] = factors is not None
            for lo, hi in [(-10, -1), (0, 1), (1, 10), (1, 20), (1, 60)]:
                label = f"CAR_{lo}_{hi}"
                event[label] = window_car(adjusted, date, lo, hi) if factors is not None else np.nan
                event[label + "_unadjusted"] = window_car(unadjusted, date, lo, hi)
            for horizon in [10, 20, 60]:
                event[f"contaminated_{horizon}"] = has_contamination(date, dates.get(code, []), calendar, horizon)
                event[f"contaminated_{horizon}"] |= same_day_periods.loc[(code, date)] > 1
            rows.append(event)
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("No strict pre-announcement matches in frozen cache")
    summary = []
    samples = {
        "strict_365d_1broker": panel.loc[panel.eligible],
        "strict_365d_2brokers": panel.loc[panel.eligible & panel.matched_brokers.ge(2)],
        "strict_180d_1broker": panel.loc[panel.eligible & panel.surprise_180.notna()].assign(surprise_pct=panel.surprise_180),
    }
    for sample_name, sample in samples.items():
        for window in ["CAR_-10_-1", "CAR_0_1", "CAR_1_10", "CAR_1_20", "CAR_1_60"]:
            horizon = max(10, int(window.rsplit("_", 1)[-1]))
            for clean in [False, True]:
                subset = sample.loc[~sample[f"contaminated_{horizon}"]] if clean else sample
                for difference in [False, True]:
                    result = clustered_estimate(subset, window, difference)
                    result.update(sample=sample_name, window=window, exclude_overlaps=clean,
                                  statistic="positive_minus_negative" if difference else "mean_car",
                                  primary_test=sample_name == "strict_365d_1broker" and window == "CAR_1_10" and clean and difference)
                    summary.append(result)
    summary = pd.DataFrame(summary)
    source = pd.DataFrame(inputs)
    if expanded:
        source.groupby("endpoint").agg(files=("path","nunique"),rows=("rows","sum"),
            empty_files=("rows",lambda values:int(values.eq(0).sum()))).to_csv(output / 'source_coverage.csv')
    if not (panel.latest_report_date < panel.ann_date).all():
        raise AssertionError("A consensus contains post-announcement information")
    if panel.duplicated(["ts_code", "period_end"]).any():
        raise AssertionError("Duplicate initial event for a stock/fiscal period")
    return_diagnostics = []
    for window in ["CAR_0_1", "CAR_1_10", "CAR_1_20", "CAR_1_60"]:
        valid = panel.loc[panel.eligible, [window, window + "_unadjusted"]].dropna()
        return_diagnostics.append({"window": window, "paired_n": len(valid),
            "unadjusted_mean": valid[window + "_unadjusted"].mean(), "adjusted_mean": valid[window].mean(),
            "events_changed_by_adjustment": int((valid[window] - valid[window + "_unadjusted"]).abs().gt(1e-8).sum())})
    funnel = pd.DataFrame([
        ("raw_forecast_rows_all_available_A_shares" if expanded else "raw_forecast_rows_300_names", forecast_rows),
        ("conflicting_same_date_rows_excluded", conflict_count),
        ("rows_in_ambiguous_fiscal_periods_excluded", int(ambiguous_period.sum())),
        ("unique_forecast_rows", len(unique)),
        ("initial_forecasts_in_event_span" if expanded else "initial_forecasts_in_price_span", len(events)),
        ("legacy_next_session_cutoff_matched", int(matched.legacy_surprise_pct.notna().sum())),
        ("legacy_latest_report_on_after_announcement", int(matched.legacy_news_contaminated.sum())),
        ("consensus_changed_after_cutoff_fix", int(matched.consensus_changed.sum())),
        ("surprise_sign_changed_after_cutoff_fix", int((matched.surprise_pct * matched.legacy_surprise_pct).lt(0).sum())),
        ("strict_announcement_cutoff_matched", len(panel)),
        ("prior_liquidity_and_listing_eligible", int(panel.eligible.sum())),
        ("matched_events_missing_liquidity_field", int((~panel.liquidity_source_available).sum())),
        ("complete_adjusted_CAR_1_10", int((panel.eligible & panel.CAR_1_10.notna()).sum())),
        ("complete_adjusted_CAR_1_10_no_overlap", int((panel.eligible & panel.CAR_1_10.notna() & ~panel.contaminated_10).sum())),
    ], columns=["stage", "count"])
    coverage = matched.groupby("period_label").agg(events=("event_id", "size"), matched=("surprise_pct", "count"))
    coverage["match_rate"] = coverage.matched / coverage.events
    if expanded:
        attributes = universe[["ts_code","exchange","market","list_status"]].rename(columns={"list_status":"listing_status_at_snapshot"})
        expanded_coverage = panel.merge(attributes,on="ts_code",validate="many_to_one")
        expanded_coverage["year"] = expanded_coverage.ann_date.dt.year
        expanded_coverage["clean_10day"] = expanded_coverage.eligible & expanded_coverage.CAR_1_10.notna() & ~expanded_coverage.contaminated_10
        for dimension in ["year","exchange","market","listing_status_at_snapshot"]:
            counts = expanded_coverage.groupby(dimension,dropna=False).agg(
                strict_events=("event_id","size"),firms=("ts_code","nunique"),
                eligible_events=("eligible","sum"),clean_10day_events=("clean_10day","sum"))
            counts.to_csv(output / f"coverage_by_{dimension}.csv")
    for name, frame in [("events", panel), ("matching_audit", matched), ("inference", summary),
                        ("sample_funnel", funnel), ("source_inventory", source), ("period_coverage", coverage.reset_index())]:
        frame.to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(return_diagnostics).to_csv(output / "return_diagnostics.csv", index=False)
    primary = summary.loc[summary.primary_test].iloc[0].to_dict()
    provenance = {"validation_date": "2026-10-04", "price_start": str(start.date()), "price_end": str(end.date()),
                  "universe": "First 300 codes in frozen current-listed stock_basic cache; not survivorship-free",
                  "forecast_unit": "CNY 10,000", "expected_np_unit": "CNY 10,000", "freshness_days": 365,
                  "day_zero": "first exchange session strictly after date-only announcement",
                  "primary_test": "positive minus negative CAR[1,10], no other earnings event in [-10,+10]",
                  "inference": "OLS two-way firm/month cluster covariance; t df=min(clusters)-1; minimum 10 clusters",
                  "report_months_possibly_capped": int(source.possible_page_cap.sum()),
                  "complete_report_refresh": complete_reports,
                  "market_data_refresh_complete": all((output / f"source_{endpoint}/receipt.json").exists() for endpoint in ["daily", "daily_basic", "fina_indicator"]),
                  "unresolved": (["Monthly report_rc caches remain capped; repair hit provider limit 1/min"] if not complete_reports else []) + ["No historical delisted universe",
                                 "No intraday release timestamps or original vintage snapshots", "Market benchmark is a price index; adjustment factors reflect corporate actions", "No trading or transaction-cost claim"]}
    if expanded:
        expansion_manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        market_receipt = json.loads((cache / "market_sources_receipt.json").read_text(encoding="utf-8"))
        report_progress_path = cache / 'report_expansion_progress.json'
        report_progress = json.loads(report_progress_path.read_text(encoding='utf-8')) if report_progress_path.exists() else {}
        provenance.update(universe=expansion_manifest["universe"], universe_size=len(universe),
            event_start=str(event_start.date()),
            normalized_report_rows=len(reports),source_forecast_rows=forecast_rows,matched_stocks=len(matched_codes),
            matched_stocks_with_formal_income_records=int(financial.ts_code.isin(matched_codes).groupby(financial.ts_code).any().sum()),
            market_source_rows=market_receipt["row_counts"],
            market_data_refresh_complete=market_receipt["complete"],
            forecast_partitions=expansion_manifest["forecast_partitions"],
            report_api_limit_observed=report_progress.get('errors', [])[-1]['error'] if report_progress.get('errors') else expansion_manifest["report_api_limit_observed"],
            unresolved=["Historical report_rc coverage remains capped/incomplete",
                        "Delisted names included, but historical code changes and source coverage are not certified complete",
                        "No intraday release timestamps or original vintage snapshots",
                        "Market benchmark is a price index; no executable trading claim"])
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    short = summary.loc[(summary["sample"] == "strict_365d_1broker") & summary.exclude_overlaps,
                        ["window", "statistic", "n", "firms", "months", "estimate", "ci_low", "ci_high", "p_value"]]
    report = f"""# 盈利意外重建审计（2026-10-04）

旧版 326 行、CAR +0.30% 的摘要撤回：它不是冻结同一数据和正确时序后得到的可复核结论。
本轮读取端点缓存，未覆盖旧原始数据/历史结果。价格实际截止 {end:%Y-%m-%d}，不是 2026-10-04。

## 已修复的问题

1. 预期截止改为公告日前，公告当天及周末后验研报均排除；同公司、同报告期、每机构最后一条有效预期再取中位数。半年/三季累计业绩不会匹配全年预测。
2. 固定交易所日历，第 0 日为公告后首个交易日；停牌/缺失不压缩窗口，不把不完整收益置零。严格区分 CAR[0,1] 与 CAR[1,10]。
3. 以 close × adj_factor 计算复权日收益；未复权版本只作诊断。保留公告后其他盈利事件污染标记，主检验排除 [-10,+10] 内其他事件。
4. 不再搜索 1,920 个规格后选择“最强”结论；主检验预先写定为清洁事件的正负意外 CAR[1,10] 差。报告双向聚类置信区间及固定敏感性组，其他检验均属探索。
5. 分页抓取报告必须读到空页，完整缓存另存 report_rc_complete；失败月份不被认证完整。修正 tp 误作目标价、快报元/万元混用、扣非净利润混作普通净利润。

## 样本流失

{funnel.to_markdown(index=False)}

报告期覆盖：

{coverage.to_markdown()}

本次所有可匹配事件都是 12 月 31 日全年报告期；其他季度的严格可用预期为零。因此当前证据只能描述年度业绩预告子样本，不能称为覆盖四个季度的盈利意外研究。

## 重算结果

以下均为算术累计市场调整收益、小数单位；例如 0.01 为 1%。CI 为 95% 双向 firm/month 聚类区间。

{short.to_markdown(index=False, floatfmt='.5f')}

主检验 N={primary['n']}，差值={primary['estimate']:.5f}，CI=[{primary['ci_low']:.5f}, {primary['ci_high']:.5f}]，p={primary['p_value']:.5f}。
这里检验的是盈利意外与后续收益的关联，不是总样本平均收益是否为正；平均 CAR 不能证明 PEAD。

同一可用事件的复权前后对照：

{pd.DataFrame(return_diagnostics).to_markdown(index=False, floatfmt='.5f')}

## 仍然限制结论的来源缺口

- 历史月度 report_rc 缓存存在截断。本轮是否已按同一 300 股票、同一日期范围逐股完整分页重抓：{complete_reports}。当前使用的疑似截断月份数为 {int(source.possible_page_cap.sum())}。本轮修复请求遇到该接口每分钟 1 次的权限限额，已停止报告刷新，未把不完整页面视为完整。完整抓取也不等于“当时可得版本”，供应商可能历史回补；重抓前的独立重算见 `frozen_cache_baseline/`。
- 股票池是当前上市股票代码排序前 300 名，存在幸存者与市场板块偏差；不代表全部 A 股。
- 公告仅有日期，采用次交易日对齐的保守约定；无盘前/盘后时间戳。快报/正式财报缓存可能遗漏后续事件，污染剔除只是已观测事件层面。
- 价格指数基准与含公司行动的个股收益口径不完全对称；双向聚类不能消除这些测量偏差。尚未验证行业/因子基准，不作市场效率或交易获利结论。
- 冻结阈值后的历史回看不等于真正前瞻样本外；180/365 天和 1/2 机构结果全部保留，不选显著项。
- 旧 PDF、演示稿和 outputs/tables 是历史产物，未同步重编；本目录是当前审计版本。

## 复现与文件

`python scripts/run_event_validation.py` 完全离线使用已保存复权因子；首次补因子用 `--refresh-factors`。
`events.csv` 是事件级账本；`matching_audit.csv` 保留失配行与修复前后预期；`sample_funnel.csv`、`period_coverage.csv`、`source_inventory.csv`、`inference.csv` 和 `provenance.json` 可逐项复核。

字段定义：[report_rc](https://tushare.pro/document/2?doc_id=292)、[forecast](https://tushare.pro/document/2?doc_id=45)、[express](https://tushare.pro/document/2?doc_id=46)、[fina_indicator](https://tushare.pro/document/2?doc_id=79)。
"""
    if expanded:
        status_counts = universe.groupby("list_status").size().to_dict()
        report = f"""# 盈利意外数据扩容（2026-10-04）

股票池由原300家扩至 {len(universe):,} 家可获取的人民币股票，包含在市、退市和暂停上市记录（状态数：{status_counts}）。价格区间 {start:%Y-%m-%d} 至 {end:%Y-%m-%d}。
旧300家冻结审计保留在 `outputs/event_validation/`。本目录仅增加来源覆盖，保持公告前同报告期匹配、365天新鲜度、1家机构、0.3%换手率及120交易日上市年龄的既定规则。

## 样本增加与流失

{funnel.to_markdown(index=False)}

{coverage.to_markdown()}

## 固定检验

以下为小数单位。主检验正负意外 CAR[1,10] 差 N={primary['n']}，估计 {primary['estimate']:.5f}，95% CI [{primary['ci_low']:.5f}, {primary['ci_high']:.5f}]，p={primary['p_value']:.5f}。清洁窗口计数包含零意外，正负组主检验排除零意外。平均CAR不等于PEAD，增加样本不保证结果更好。

{short.to_markdown(index=False, floatfmt='.5f')}

## 覆盖与限制

- 使用截至本轮可获得的全部股票名录，包括退市公司，并按事件日上市/退市日期判断资格；仍无法认证历史代码迁移与供应商历史覆盖完整。
- 业绩预告、快报通过普通接口按自然公告日分页重新取得（含周末）；VIP接口无权限。正式财报按匹配股票补齐，同时保留公告日和实际公告日用于重叠事件筛查。来源逐分片留有回执。
- 研报仍含 {int(source.possible_page_cap.sum())} 份可能截断的旧月度缓存。扩容时接口进一步返回每天10次额度已用尽，已停止请求；未将被截断月份认证为全量。新增完整分片与旧缓存合并去重，报告缺失仍影响覆盖；未放宽同报告期匹配。
- 行情、换手率、复权因子和利润表的股票级请求均已完成；其中 {int((source.endpoint.eq('adj_factor') & source.rows.eq(0)).sum())} 份复权快照为空，请求完成不代表供应商数据无缺失。停牌、缺失、未成熟收益窗口保留缺失，不填零。
- 当日未知公告时刻、历史版本修订、价格指数基准口径等限制仍在，不作市场整体效率或交易获利结论。

复现：`python scripts/run_event_validation.py --expanded`。扩容准备与断点续取见 `scripts/expand_event_data.py`，授权原始数据仅存本地 `data_raw/expansion_20260930/`。`source_inventory.csv`、`provenance.json`、`sample_funnel.csv`和事件级账本记录口径。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(funnel.to_string(index=False))
    print(short.to_string(index=False))
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-factors", action="store_true")
    parser.add_argument("--matching-only", action="store_true", help="Prepare the exact matched universe before bounded source refresh")
    parser.add_argument("--expanded", action="store_true", help="Use expanded all-A-share snapshots; preserve the 300-stock baseline")
    run(**vars(parser.parse_args()))
