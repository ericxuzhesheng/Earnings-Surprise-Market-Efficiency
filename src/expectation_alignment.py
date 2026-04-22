from __future__ import annotations

from collections.abc import Iterable
import re

import numpy as np
import pandas as pd

from src.config import ProjectConfig


BENCHMARK_FIELD_MAP = {
    "np": "np",
    "eps": "eps",
    "target_price": "target_price_mid",
    "np_first": "np",
}

MATCH_TIERS = (
    "strict_same_quarter",
    "same_fiscal_year_nearest_valid",
    "latest_valid_pre_event",
    "multi_report_median",
)

MATCH_TIER_ALIASES = {
    "strict_pre_event_same_period": "strict_same_quarter",
    "nearest_valid_public_expectation": "same_fiscal_year_nearest_valid",
    "text_proxy_only": "multi_report_median",
}

REPORT_RC_SOURCE_NAME = "tushare_report_rc"
REPORT_RC_SOURCE_TIER = "tier_1_tushare_report_rc"
REPORT_RC_SOURCE_FLAGS = {
    "source_name": REPORT_RC_SOURCE_NAME,
    "source_tier": REPORT_RC_SOURCE_TIER,
    "expectation_tier": REPORT_RC_SOURCE_TIER,
    "is_official_source": 0,
    "is_aggregated_source": 1,
    "is_text_proxy": 0,
}

EASTMONEY_FORECAST_SOURCE_NAME = "eastmoney_profit_forecast"
EASTMONEY_FORECAST_SOURCE_TIER = "tier_2_eastmoney_profit_forecast"
EASTMONEY_FORECAST_FLAGS = {
    "source_name": EASTMONEY_FORECAST_SOURCE_NAME,
    "source_tier": EASTMONEY_FORECAST_SOURCE_TIER,
    "expectation_tier": EASTMONEY_FORECAST_SOURCE_TIER,
    "is_official_source": 0,
    "is_aggregated_source": 1,
    "is_text_proxy": 0,
}

EASTMONEY_RESEARCH_SOURCE_NAME = "eastmoney_research_report"
EASTMONEY_RESEARCH_SOURCE_TIER = "tier_3_eastmoney_research_report_text"
EASTMONEY_RESEARCH_FLAGS = {
    "source_name": EASTMONEY_RESEARCH_SOURCE_NAME,
    "source_tier": EASTMONEY_RESEARCH_SOURCE_TIER,
    "expectation_tier": EASTMONEY_RESEARCH_SOURCE_TIER,
    "is_official_source": 0,
    "is_aggregated_source": 0,
    "is_text_proxy": 1,
}

EXPECTED_SCHEMA_COLUMNS = [
    "ts_code",
    "period_end",
    "report_date",
    "fiscal_year",
    "analyst_entity",
    "org_name",
    "author_name",
    "np",
    "eps",
    "pe",
    "target_price_mid",
    "rating",
    "title_over_expectation_flag",
    "matched_report_count_proxy",
    "source_name",
    "source_tier",
    "expectation_tier",
    "is_official_source",
    "is_aggregated_source",
    "is_text_proxy",
]


def _consensus_column(config: ProjectConfig) -> str:
    return BENCHMARK_FIELD_MAP.get(config.consensus_value_field, "np")


def _normalize_consensus_method(method: str) -> str:
    normalized = (method or "").strip().lower()
    mapping = {
        "latest": "latest_snapshot",
        "latest_snapshot": "latest_snapshot",
        "latest_per_analyst": "latest_per_analyst",
        "mean": "pooled_mean",
        "median": "pooled_median",
        "pooled_mean": "pooled_mean",
        "pooled_median": "pooled_median",
    }
    return mapping.get(normalized, "pooled_median")


def _rating_value(series: pd.Series) -> float | str | None:
    clean = series.dropna()
    return clean.iloc[-1] if not clean.empty else np.nan


def _add_entity_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    org = out.get("org_name", pd.Series(index=out.index, dtype=object)).fillna("").astype(str).str.strip()
    author = out.get("author_name", pd.Series(index=out.index, dtype=object)).fillna("").astype(str).str.strip()
    entity = org.mask(org.eq(""), author).mask(lambda s: s.eq(""), "unknown_analyst")
    out["analyst_entity"] = entity
    return out


def _normalize_match_tier(tier: str) -> str:
    normalized = (tier or "").strip().lower()
    normalized = MATCH_TIER_ALIASES.get(normalized, normalized)
    if normalized in MATCH_TIERS:
        return normalized
    return "strict_same_quarter"


def _tier_group(tier: str) -> str:
    return "strict" if _normalize_match_tier(tier) == "strict_same_quarter" else "relaxed"


def _empty_expectation_schema() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPECTED_SCHEMA_COLUMNS)


def _extract_year_from_text(value: object) -> float:
    text = str(value or "").strip()
    match = re.search(r"(20\d{2})", text)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except ValueError:
        return np.nan


def _prepare_expectation_source(
    df: pd.DataFrame,
    source_name: str,
    source_tier: str,
    is_official_source: int,
    is_aggregated_source: int,
    is_text_proxy: int,
) -> pd.DataFrame:
    if df.empty:
        return _empty_expectation_schema()
    out = _add_entity_key(df)
    out["ts_code"] = out.get("ts_code", pd.Series(index=out.index, dtype=object)).astype(str)
    out["period_end"] = pd.to_datetime(out.get("period_end"), errors="coerce")
    out["report_date"] = pd.to_datetime(out.get("report_date"), errors="coerce")
    out["fiscal_year"] = pd.to_numeric(out.get("fiscal_year"), errors="coerce")
    for col in ["np", "eps", "pe", "target_price_mid", "matched_report_count_proxy"]:
        out[col] = pd.to_numeric(out.get(col, np.nan), errors="coerce")
    out["title_over_expectation_flag"] = pd.to_numeric(
        out.get("title_over_expectation_flag", 0), errors="coerce"
    ).fillna(0)
    for col in ["org_name", "author_name", "rating"]:
        if col not in out.columns:
            out[col] = np.nan
    out["source_name"] = out.get("source_name", source_name).fillna(source_name)
    out["source_tier"] = out.get("source_tier", source_tier).fillna(source_tier)
    out["expectation_tier"] = out.get("expectation_tier", source_tier).fillna(source_tier)
    out["is_official_source"] = pd.to_numeric(
        out.get("is_official_source", is_official_source), errors="coerce"
    ).fillna(is_official_source).astype(int)
    out["is_aggregated_source"] = pd.to_numeric(
        out.get("is_aggregated_source", is_aggregated_source), errors="coerce"
    ).fillna(is_aggregated_source).astype(int)
    out["is_text_proxy"] = pd.to_numeric(
        out.get("is_text_proxy", is_text_proxy), errors="coerce"
    ).fillna(is_text_proxy).astype(int)
    return out[EXPECTED_SCHEMA_COLUMNS].copy()


def _prepare_report_rc_expectations(report_rc_df: pd.DataFrame) -> pd.DataFrame:
    return _prepare_expectation_source(
        df=report_rc_df,
        source_name=REPORT_RC_SOURCE_NAME,
        source_tier=REPORT_RC_SOURCE_TIER,
        is_official_source=REPORT_RC_SOURCE_FLAGS["is_official_source"],
        is_aggregated_source=REPORT_RC_SOURCE_FLAGS["is_aggregated_source"],
        is_text_proxy=REPORT_RC_SOURCE_FLAGS["is_text_proxy"],
    )


def _prepare_eastmoney_profit_forecast_expectations(expectation_df: pd.DataFrame) -> pd.DataFrame:
    if expectation_df is None or expectation_df.empty:
        return _empty_expectation_schema()
    out = expectation_df.copy()
    out["report_date"] = pd.to_datetime(out.get("report_date"), errors="coerce")
    if out["report_date"].isna().all():
        out["report_date"] = pd.Timestamp.today().normalize()
    if "eps" not in out.columns:
        out["eps"] = pd.to_numeric(out.get("forecast_eps_latest", np.nan), errors="coerce")
    out["np"] = np.nan
    out["fiscal_year"] = pd.to_numeric(out.get("fiscal_year"), errors="coerce")
    if out["fiscal_year"].isna().all():
        out["fiscal_year"] = out["report_date"].dt.year
    out["period_end"] = pd.to_datetime(
        dict(year=pd.to_numeric(out["fiscal_year"], errors="coerce"), month=12, day=31),
        errors="coerce",
    )
    out["analyst_entity"] = out.get("analyst_entity", EASTMONEY_FORECAST_SOURCE_NAME)
    out["org_name"] = out.get("org_name", np.nan)
    out["author_name"] = out.get("author_name", np.nan)
    out["matched_report_count_proxy"] = pd.to_numeric(
        out.get("matched_report_count_proxy", out.get("研报数", out.get("研报数量", np.nan))),
        errors="coerce",
    )
    out["pe"] = pd.to_numeric(out.get("pe", np.nan), errors="coerce")
    out["target_price_mid"] = pd.to_numeric(out.get("target_price_mid", np.nan), errors="coerce")
    out["rating"] = out.get("rating", np.nan)
    out["title_over_expectation_flag"] = pd.to_numeric(
        out.get("title_over_expectation_flag", 0), errors="coerce"
    ).fillna(0)
    return _prepare_expectation_source(
        df=out,
        source_name=EASTMONEY_FORECAST_SOURCE_NAME,
        source_tier=EASTMONEY_FORECAST_SOURCE_TIER,
        is_official_source=EASTMONEY_FORECAST_FLAGS["is_official_source"],
        is_aggregated_source=EASTMONEY_FORECAST_FLAGS["is_aggregated_source"],
        is_text_proxy=EASTMONEY_FORECAST_FLAGS["is_text_proxy"],
    )


def _prepare_eastmoney_research_expectations(report_df: pd.DataFrame) -> pd.DataFrame:
    if report_df is None or report_df.empty:
        return _empty_expectation_schema()
    out = report_df.copy()
    out["report_date"] = pd.to_datetime(out.get("report_date"), errors="coerce")
    out["fiscal_year"] = pd.to_numeric(out.get("fiscal_year"), errors="coerce")
    if out["fiscal_year"].isna().all():
        out["fiscal_year"] = out["report_date"].dt.year
    eps_cols = [c for c in out.columns if "预测-收益" in str(c) or "预测每股收益" in str(c)]
    if "eps" not in out.columns:
        out["eps"] = np.nan
        for col in eps_cols:
            candidate = pd.to_numeric(out[col], errors="coerce")
            if candidate.notna().any():
                out["eps"] = candidate
                break
    out["period_end"] = pd.to_datetime(
        dict(year=pd.to_numeric(out["fiscal_year"], errors="coerce"), month=12, day=31),
        errors="coerce",
    )
    out["np"] = np.nan
    out["analyst_entity"] = out.get(
        "analyst_entity",
        out.get("org_name", out.get("orgSName", EASTMONEY_RESEARCH_SOURCE_NAME)),
    )
    out["org_name"] = out.get("org_name", out.get("orgSName", np.nan))
    out["author_name"] = out.get("author_name", out.get("研究员", out.get("作者", np.nan)))
    out["matched_report_count_proxy"] = pd.to_numeric(out.get("matched_report_count_proxy", 1), errors="coerce")
    out["pe"] = pd.to_numeric(out.get("pe", np.nan), errors="coerce")
    out["target_price_mid"] = pd.to_numeric(out.get("target_price_mid", np.nan), errors="coerce")
    out["rating"] = out.get("rating", out.get("评级", np.nan))
    out["title_over_expectation_flag"] = pd.to_numeric(
        out.get("title_over_expectation_flag", 0), errors="coerce"
    ).fillna(0)
    return _prepare_expectation_source(
        df=out,
        source_name=EASTMONEY_RESEARCH_SOURCE_NAME,
        source_tier=EASTMONEY_RESEARCH_SOURCE_TIER,
        is_official_source=EASTMONEY_RESEARCH_FLAGS["is_official_source"],
        is_aggregated_source=EASTMONEY_RESEARCH_FLAGS["is_aggregated_source"],
        is_text_proxy=EASTMONEY_RESEARCH_FLAGS["is_text_proxy"],
    )


def build_expectation_panel(report_rc_df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if report_rc_df.empty:
        return pd.DataFrame()
    df = _add_entity_key(report_rc_df)
    value_col = _consensus_column(config)
    if value_col not in df.columns:
        df[value_col] = np.nan
    agg = (
        df.groupby(["ts_code", "period_end", "report_date"], as_index=False)
        .agg(
            consensus_np=("np", "median"),
            consensus_eps=("eps", "median"),
            consensus_pe=("pe", "median"),
            consensus_target_price=("target_price_mid", "median"),
            consensus_rating=("rating", _rating_value),
            report_count=(value_col, lambda s: int(s.notna().sum())),
            broker_count=("analyst_entity", lambda s: int(s.dropna().nunique())),
            title_over_expectation_count=("title_over_expectation_flag", "sum"),
        )
        .sort_values(["ts_code", "period_end", "report_date"])
        .reset_index(drop=True)
    )
    agg["fraction_title_over_expectation"] = np.where(
        agg["report_count"] > 0,
        agg["title_over_expectation_count"] / agg["report_count"],
        np.nan,
    )
    return agg


def _prepare_source_groups(expectation_df: pd.DataFrame) -> dict[object, pd.DataFrame]:
    if expectation_df.empty:
        return {}
    groups: dict[object, pd.DataFrame] = {}
    for ts_code, grp in expectation_df.groupby("ts_code", dropna=False):
        groups[ts_code] = grp.sort_values(["period_end", "report_date", "analyst_entity"]).reset_index(drop=True)
    return groups


def _source_label_from_tier(source_tier: str) -> str:
    mapping = {
        REPORT_RC_SOURCE_TIER: REPORT_RC_SOURCE_NAME,
        EASTMONEY_FORECAST_SOURCE_TIER: EASTMONEY_FORECAST_SOURCE_NAME,
        EASTMONEY_RESEARCH_SOURCE_TIER: EASTMONEY_RESEARCH_SOURCE_NAME,
    }
    return mapping.get(source_tier, str(source_tier or "unknown_source"))


def _select_consensus_candidates(candidates: pd.DataFrame, method: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    method = _normalize_consensus_method(method)
    ordered = candidates.sort_values(["report_date", "analyst_entity"]).copy()
    if method == "latest_snapshot":
        latest_date = ordered["report_date"].max()
        return ordered[ordered["report_date"] == latest_date].copy()
    if method == "latest_per_analyst":
        return ordered.drop_duplicates(subset=["analyst_entity"], keep="last").copy()
    return ordered.copy()


def _default_match_values(
    config: ProjectConfig,
    method: str,
    tier: str,
    source_name: str = REPORT_RC_SOURCE_NAME,
    source_tier: str = REPORT_RC_SOURCE_TIER,
    is_official_source: int = 0,
    is_aggregated_source: int = 1,
    is_text_proxy: int = 0,
) -> dict[str, object]:
    benchmark_field = _consensus_column(config)
    normalized_method = _normalize_consensus_method(method)
    normalized_tier = _normalize_match_tier(tier)
    tier_group = _tier_group(normalized_tier)
    return {
        "benchmark_method": normalized_method,
        "benchmark_value_field": benchmark_field,
        "expectation_source": f"missing_{source_name}",
        "expectation_source_name": source_name,
        "expectation_source_tier": source_tier,
        "source_name": source_name,
        "source_tier": source_tier,
        "expectation_tier": source_tier,
        "match_quality": "missing",
        "match_tier": normalized_tier,
        "match_tier_group": tier_group,
        "matched_report_date": pd.NaT,
        "matched_report_count": 0,
        "matched_broker_count": 0,
        "report_rc_match_quality": "missing",
        "report_rc_match_tier": normalized_tier,
        "report_rc_match_tier_group": tier_group,
        "benchmark_lag_days": np.nan,
        "candidate_report_count_total": 0,
        "candidate_broker_count_total": 0,
        "selected_candidate_rows": 0,
        "benchmark_value_count": 0,
        "expected_np": np.nan,
        "expected_eps": np.nan,
        "expected_target_price": np.nan,
        "expected_rating": np.nan,
        "text_over_expectation_fraction": np.nan,
        "benchmark_has_np": 0,
        "benchmark_has_eps": 0,
        "is_official_source": int(is_official_source),
        "is_aggregated_source": int(is_aggregated_source),
        "is_text_proxy": int(is_text_proxy),
    }


def _aggregate_selected_candidates(
    selected: pd.DataFrame,
    event_date: pd.Timestamp,
    config: ProjectConfig,
    method: str,
    tier: str,
    source_name: str,
    source_tier: str,
    is_official_source: int,
    is_aggregated_source: int,
    is_text_proxy: int,
) -> dict[str, object]:
    defaults = _default_match_values(
        config=config,
        method=method,
        tier=tier,
        source_name=source_name,
        source_tier=source_tier,
        is_official_source=is_official_source,
        is_aggregated_source=is_aggregated_source,
        is_text_proxy=is_text_proxy,
    )
    benchmark_field = str(defaults["benchmark_value_field"])
    if selected.empty:
        return defaults

    agg_fn = "mean" if defaults["benchmark_method"] == "pooled_mean" else "median"
    expected_np = selected["np"].mean() if agg_fn == "mean" else selected["np"].median()
    expected_eps = selected["eps"].mean() if agg_fn == "mean" else selected["eps"].median()
    expected_target_price = selected["target_price_mid"].mean() if agg_fn == "mean" else selected["target_price_mid"].median()
    text_fraction = selected["title_over_expectation_flag"].mean() if agg_fn == "mean" else selected["title_over_expectation_flag"].median()

    matched_report_date = selected["report_date"].max()
    matched_report_count = int(selected[benchmark_field].notna().sum()) if benchmark_field in selected.columns else 0
    matched_broker_count = int(selected.loc[selected[benchmark_field].notna(), "analyst_entity"].nunique()) if benchmark_field in selected.columns else 0
    min_count_ok = matched_report_count >= config.min_valid_report_count
    lag_days = (event_date - matched_report_date).days if pd.notna(matched_report_date) else np.nan
    match_quality = "strict" if min_count_ok else "weak_count"

    return {
        **defaults,
        "expectation_source": source_name if min_count_ok else f"{source_name}_below_min_count",
        "match_quality": match_quality,
        "matched_report_date": matched_report_date,
        "matched_report_count": matched_report_count,
        "matched_broker_count": matched_broker_count,
        "report_rc_match_quality": match_quality,
        "benchmark_lag_days": lag_days,
        "candidate_report_count_total": matched_report_count,
        "candidate_broker_count_total": int(selected["analyst_entity"].nunique()),
        "selected_candidate_rows": int(len(selected)),
        "benchmark_value_count": matched_report_count,
        "expected_np": expected_np,
        "expected_eps": expected_eps,
        "expected_target_price": expected_target_price,
        "expected_rating": _rating_value(selected["rating"]),
        "text_over_expectation_fraction": text_fraction,
        "benchmark_has_np": int(selected["np"].notna().any()),
        "benchmark_has_eps": int(selected["eps"].notna().any()),
    }


def _method_for_tier(method: str, tier: str) -> str:
    if _normalize_match_tier(tier) == "multi_report_median":
        return "pooled_median"
    return _normalize_consensus_method(method)


def _base_candidates_for_event(group: pd.DataFrame, event_date: pd.Timestamp, freshness_days: int) -> pd.DataFrame:
    freshness = pd.Timedelta(days=freshness_days)
    return group[(group["report_date"] < event_date) & (group["report_date"] >= event_date - freshness)].copy()


def _nearest_period_subset(candidates: pd.DataFrame, event_period_end: pd.Timestamp, benchmark_field: str) -> pd.DataFrame:
    if candidates.empty or pd.isna(event_period_end):
        return pd.DataFrame(columns=candidates.columns)
    valid = candidates[candidates[benchmark_field].notna()].copy() if benchmark_field in candidates.columns else candidates.copy()
    if valid.empty:
        return pd.DataFrame(columns=candidates.columns)
    unique_periods = valid["period_end"].dropna().drop_duplicates()
    if unique_periods.empty:
        return pd.DataFrame(columns=candidates.columns)
    distance = (unique_periods - event_period_end).abs()
    nearest_period = unique_periods.loc[distance.sort_values().index].iloc[0]
    return candidates[candidates["period_end"] == nearest_period].copy()


def _latest_period_subset(candidates: pd.DataFrame, benchmark_field: str) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=candidates.columns)
    valid = candidates[candidates[benchmark_field].notna()].copy() if benchmark_field in candidates.columns else candidates.copy()
    if valid.empty:
        return pd.DataFrame(columns=candidates.columns)
    latest_period = valid["period_end"].dropna().max()
    if pd.isna(latest_period):
        latest_report_date = valid["report_date"].max()
        return valid[valid["report_date"] == latest_report_date].copy()
    return candidates[candidates["period_end"] == latest_period].copy()


def _candidate_universe_for_tier(event: pd.Series, group: pd.DataFrame, config: ProjectConfig, tier: str) -> pd.DataFrame:
    benchmark_field = _consensus_column(config)
    event_date = pd.to_datetime(event.get("event_trade_date"), errors="coerce")
    event_period_end = pd.to_datetime(event.get("period_end"), errors="coerce")
    base = _base_candidates_for_event(group=group, event_date=event_date, freshness_days=config.report_freshness_days)
    if base.empty:
        return base

    tier = _normalize_match_tier(tier)
    if tier == "strict_same_quarter":
        return base[base["period_end"] == event_period_end].copy()
    if tier == "same_fiscal_year_nearest_valid":
        if pd.isna(event_period_end):
            return pd.DataFrame(columns=base.columns)
        event_year = event_period_end.year
        same_year = base[base.get("fiscal_year", pd.Series(index=base.index, dtype=float)).eq(event_year)].copy()
        return _nearest_period_subset(same_year, event_period_end=event_period_end, benchmark_field=benchmark_field)
    if tier == "latest_valid_pre_event":
        return _latest_period_subset(base, benchmark_field=benchmark_field)
    return base[base[benchmark_field].notna()].copy() if benchmark_field in base.columns else base.copy()


def _candidate_rows_for_event(
    event: pd.Series,
    candidates: pd.DataFrame,
    selected_index: Iterable[object],
    benchmark_method: str,
    match_tier: str,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    out = candidates.copy()
    out["event_id"] = event.get("event_id")
    out["event_type"] = event.get("event_type")
    out["event_trade_date"] = pd.to_datetime(event.get("event_trade_date"))
    out["period_end"] = pd.to_datetime(event.get("period_end"))
    normalized_tier = _normalize_match_tier(match_tier)
    tier_group = _tier_group(match_tier)
    out["benchmark_method"] = benchmark_method
    out["match_tier"] = normalized_tier
    out["match_tier_group"] = tier_group
    out["report_rc_match_tier"] = normalized_tier
    out["report_rc_match_tier_group"] = tier_group
    out["expectation_source_name"] = out.get("source_name", np.nan)
    out["expectation_source_tier"] = out.get("source_tier", np.nan)
    out["source_name"] = out.get("source_name", np.nan)
    out["source_tier"] = out.get("source_tier", np.nan)
    out["expectation_tier"] = out.get("expectation_tier", out.get("source_tier", np.nan))
    out["is_selected_for_benchmark"] = out.index.isin(list(selected_index))
    out["match_quality"] = np.where(out["is_selected_for_benchmark"], "selected_candidate", "candidate_only")
    latest_date = out["report_date"].max()
    out["is_latest_snapshot"] = out["report_date"] == latest_date
    latest_per_entity = out.groupby("analyst_entity")["report_date"].transform("max")
    out["is_latest_per_analyst"] = out["report_date"] == latest_per_entity
    out["lag_days"] = (out["event_trade_date"] - out["report_date"]).dt.days
    keep_cols = [
        "event_id",
        "ts_code",
        "event_type",
        "event_trade_date",
        "period_end",
        "benchmark_method",
        "match_tier",
        "match_tier_group",
        "report_rc_match_tier",
        "report_rc_match_tier_group",
        "expectation_source_name",
        "expectation_source_tier",
        "source_name",
        "source_tier",
        "expectation_tier",
        "match_quality",
        "is_official_source",
        "is_aggregated_source",
        "is_text_proxy",
        "report_date",
        "lag_days",
        "analyst_entity",
        "org_name",
        "author_name",
        "np",
        "eps",
        "pe",
        "target_price_mid",
        "rating",
        "title_over_expectation_flag",
        "is_latest_snapshot",
        "is_latest_per_analyst",
        "is_selected_for_benchmark",
    ]
    existing = [c for c in keep_cols if c in out.columns]
    return out[existing].copy()


def _source_stack_from_config(config: ProjectConfig, available_tiers: set[str]) -> list[str]:
    configured = [tier.strip() for tier in config.source_tier_stack.split(",") if tier.strip()]
    ordered = [tier for tier in configured if tier in available_tiers]
    for fallback in [REPORT_RC_SOURCE_TIER, EASTMONEY_FORECAST_SOURCE_TIER, EASTMONEY_RESEARCH_SOURCE_TIER]:
        if fallback in available_tiers and fallback not in ordered:
            ordered.append(fallback)
    return ordered


def _choose_best_source_match(source_matches: list[dict[str, object]], fallback_source_tier: str) -> dict[str, object]:
    for match in source_matches:
        if int(match.get("benchmark_value_count", 0) or 0) > 0:
            return match
    for match in source_matches:
        if match.get("source_tier") == fallback_source_tier:
            return match
    return source_matches[0] if source_matches else {}


def match_expectations_to_events(
    events_df: pd.DataFrame,
    report_rc_df: pd.DataFrame,
    config: ProjectConfig,
    selected_tier: str = "strict_same_quarter",
    eastmoney_profit_forecast_df: pd.DataFrame | None = None,
    eastmoney_research_report_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_tier = _normalize_match_tier(selected_tier)
    if events_df.empty:
        return events_df.copy(), pd.DataFrame(), pd.DataFrame()

    prepared_sources = {
        REPORT_RC_SOURCE_TIER: _prepare_report_rc_expectations(report_rc_df),
        EASTMONEY_FORECAST_SOURCE_TIER: _prepare_eastmoney_profit_forecast_expectations(
            eastmoney_profit_forecast_df if eastmoney_profit_forecast_df is not None else pd.DataFrame()
        ),
        EASTMONEY_RESEARCH_SOURCE_TIER: _prepare_eastmoney_research_expectations(
            eastmoney_research_report_df if eastmoney_research_report_df is not None else pd.DataFrame()
        ),
    }
    source_groups = {tier: _prepare_source_groups(df) for tier, df in prepared_sources.items()}
    available_tiers = {tier for tier, df in prepared_sources.items() if not df.empty}
    source_stack = _source_stack_from_config(config, available_tiers or {REPORT_RC_SOURCE_TIER})
    default_method = _normalize_consensus_method(config.consensus_method)

    matched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []

    for _, event in events_df.iterrows():
        event_date = pd.to_datetime(event.get("event_trade_date"), errors="coerce")
        period_end = pd.to_datetime(event.get("period_end"), errors="coerce")
        tier_matches: dict[str, dict[str, object]] = {}

        for tier in MATCH_TIERS:
            method = _method_for_tier(default_method, tier)
            source_matches: list[dict[str, object]] = []

            for source_tier in source_stack:
                group = source_groups.get(source_tier, {}).get(event.get("ts_code"))
                source_name = _source_label_from_tier(source_tier)
                if group is None or group.empty:
                    match = _default_match_values(
                        config=config,
                        method=method,
                        tier=tier,
                        source_name=source_name,
                        source_tier=source_tier,
                        is_official_source=0,
                        is_aggregated_source=1 if source_tier != EASTMONEY_RESEARCH_SOURCE_TIER else 0,
                        is_text_proxy=1 if source_tier == EASTMONEY_RESEARCH_SOURCE_TIER else 0,
                    )
                    selected = pd.DataFrame()
                    candidates = pd.DataFrame()
                else:
                    candidates = _candidate_universe_for_tier(event=event, group=group, config=config, tier=tier)
                    candidates = candidates.copy()
                    if not candidates.empty:
                        candidates["lag_days"] = (event_date - candidates["report_date"]).dt.days
                    selected = _select_consensus_candidates(candidates, method) if not candidates.empty else pd.DataFrame()
                    match = _aggregate_selected_candidates(
                        selected=selected,
                        event_date=event_date,
                        config=config,
                        method=method,
                        tier=tier,
                        source_name=str(candidates.get("source_name", pd.Series([source_name])).iloc[0]) if not candidates.empty else source_name,
                        source_tier=str(candidates.get("source_tier", pd.Series([source_tier])).iloc[0]) if not candidates.empty else source_tier,
                        is_official_source=int(candidates.get("is_official_source", pd.Series([0])).iloc[0]) if not candidates.empty else 0,
                        is_aggregated_source=int(candidates.get("is_aggregated_source", pd.Series([0])).iloc[0]) if not candidates.empty else 0,
                        is_text_proxy=int(candidates.get("is_text_proxy", pd.Series([0])).iloc[0]) if not candidates.empty else 0,
                    )
                    benchmark_field = _consensus_column(config)
                    match["candidate_report_count_total"] = int(candidates[benchmark_field].notna().sum()) if benchmark_field in candidates.columns else 0
                    match["candidate_broker_count_total"] = int(candidates.loc[candidates[benchmark_field].notna(), "analyst_entity"].nunique()) if benchmark_field in candidates.columns else 0
                    if not candidates.empty:
                        candidate_frames.append(
                            _candidate_rows_for_event(event, candidates, selected.index, match["benchmark_method"], tier)
                        )

                match["source_stack_position"] = source_stack.index(source_tier) + 1 if source_tier in source_stack else np.nan
                match["selected_for_event_match"] = 0
                source_matches.append(match)

            chosen_match = _choose_best_source_match(source_matches, fallback_source_tier=REPORT_RC_SOURCE_TIER)
            chosen_match = chosen_match.copy()
            chosen_match["selected_for_event_match"] = 1
            tier_matches[tier] = chosen_match

            for match in source_matches:
                audit_rows.append(
                    {
                        "event_id": event.get("event_id"),
                        "ts_code": event.get("ts_code"),
                        "event_type": event.get("event_type"),
                        "event_trade_date": event_date,
                        "period_end": period_end,
                        **match,
                    }
                )

        row = event.to_dict()
        row.update(tier_matches.get(selected_tier, _default_match_values(config, default_method, selected_tier)))
        matched_rows.append(row)

    matched_df = pd.DataFrame(matched_rows)
    audit_df = pd.DataFrame(audit_rows)
    candidate_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    return matched_df, audit_df, candidate_df


def build_sell_side_revision_panel(report_rc_df: pd.DataFrame) -> pd.DataFrame:
    if report_rc_df.empty:
        return pd.DataFrame()
    df = _add_entity_key(report_rc_df)
    df = df.sort_values(["ts_code", "period_end", "analyst_entity", "report_date"]).reset_index(drop=True)
    for col in ["np", "eps", "target_price_mid"]:
        if col in df.columns:
            df[f"prev_{col}"] = df.groupby(["ts_code", "period_end", "analyst_entity"])[col].shift(1)
            df[f"delta_{col}"] = df[col] - df[f"prev_{col}"]
    rev = (
        df.groupby(["ts_code", "period_end", "report_date"], as_index=False)
        .agg(
            revision_magnitude_np=("delta_np", "median"),
            revision_magnitude_eps=("delta_eps", "median"),
            target_price_change=("delta_target_price_mid", "median"),
            upward_revision_count=("delta_np", lambda s: int((s > 0).sum())),
            downward_revision_count=("delta_np", lambda s: int((s < 0).sum())),
            analyst_count=("analyst_entity", lambda s: int(s.dropna().nunique())),
        )
        .sort_values(["ts_code", "period_end", "report_date"])
        .reset_index(drop=True)
    )
    rev["fraction_upgraded"] = np.where(
        rev["analyst_count"] > 0,
        rev["upward_revision_count"] / rev["analyst_count"],
        np.nan,
    )
    return rev
