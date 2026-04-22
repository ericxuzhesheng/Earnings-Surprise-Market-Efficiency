from __future__ import annotations

from collections.abc import Iterable

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
    if normalized in MATCH_TIERS:
        return normalized
    return "strict_same_quarter"


def _tier_group(tier: str) -> str:
    tier = _normalize_match_tier(tier)
    if tier == "strict_same_quarter":
        return "strict"
    return "relaxed"


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


def _prepare_report_groups(report_rc_df: pd.DataFrame) -> dict[object, pd.DataFrame]:
    if report_rc_df.empty:
        return {}
    df = _add_entity_key(report_rc_df)
    df["period_end"] = pd.to_datetime(df.get("period_end"), errors="coerce")
    df["report_date"] = pd.to_datetime(df.get("report_date"), errors="coerce")
    groups: dict[object, pd.DataFrame] = {}
    for ts_code, grp in df.groupby("ts_code", dropna=False):
        groups[ts_code] = grp.sort_values(["period_end", "report_date", "analyst_entity"]).reset_index(drop=True)
    return groups


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
    if method == "pooled_mean":
        return ordered.copy()
    return ordered.copy()


def _default_match_values(config: ProjectConfig, method: str, tier: str) -> dict[str, object]:
    benchmark_field = _consensus_column(config)
    normalized_method = _normalize_consensus_method(method)
    normalized_tier = _normalize_match_tier(tier)
    return {
        "benchmark_method": normalized_method,
        "benchmark_value_field": benchmark_field,
        "expectation_source": "missing_report_rc",
        "matched_report_date": pd.NaT,
        "matched_report_count": 0,
        "matched_broker_count": 0,
        "report_rc_match_quality": "missing",
        "report_rc_match_tier": normalized_tier,
        "report_rc_match_tier_group": _tier_group(normalized_tier),
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
    }


def _aggregate_selected_candidates(
    selected: pd.DataFrame,
    event_date: pd.Timestamp,
    config: ProjectConfig,
    method: str,
    tier: str,
) -> dict[str, object]:
    defaults = _default_match_values(config=config, method=method, tier=tier)
    benchmark_field = str(defaults["benchmark_value_field"])
    if selected.empty:
        return defaults

    agg_fn = "mean" if defaults["benchmark_method"] == "pooled_mean" else "median"
    if agg_fn == "mean":
        expected_np = selected["np"].mean()
        expected_eps = selected["eps"].mean()
        expected_target_price = selected["target_price_mid"].mean()
        text_fraction = selected["title_over_expectation_flag"].mean()
    else:
        expected_np = selected["np"].median()
        expected_eps = selected["eps"].median()
        expected_target_price = selected["target_price_mid"].median()
        text_fraction = selected["title_over_expectation_flag"].median()

    matched_report_date = selected["report_date"].max()
    matched_report_count = int(selected[benchmark_field].notna().sum()) if benchmark_field in selected.columns else 0
    matched_broker_count = int(selected.loc[selected[benchmark_field].notna(), "analyst_entity"].nunique()) if benchmark_field in selected.columns else 0
    min_count_ok = matched_report_count >= config.min_valid_report_count
    lag_days = (event_date - matched_report_date).days if pd.notna(matched_report_date) else np.nan

    return {
        **defaults,
        "expectation_source": "report_rc" if min_count_ok else "report_rc_below_min_count",
        "matched_report_date": matched_report_date,
        "matched_report_count": matched_report_count,
        "matched_broker_count": matched_broker_count,
        "report_rc_match_quality": "strict" if min_count_ok else "weak_count",
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


def _base_candidates_for_event(
    group: pd.DataFrame,
    event_date: pd.Timestamp,
    freshness_days: int,
) -> pd.DataFrame:
    freshness = pd.Timedelta(days=freshness_days)
    return group[
        (group["report_date"] < event_date)
        & (group["report_date"] >= event_date - freshness)
    ].copy()


def _nearest_period_subset(
    candidates: pd.DataFrame,
    event_period_end: pd.Timestamp,
    benchmark_field: str,
) -> pd.DataFrame:
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


def _candidate_universe_for_tier(
    event: pd.Series,
    group: pd.DataFrame,
    config: ProjectConfig,
    tier: str,
) -> pd.DataFrame:
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
        same_year = base[
            base.get("fiscal_year", pd.Series(index=base.index, dtype=float)).eq(event_year)
            & base["period_end"].ne(event_period_end)
        ].copy()
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
    out["benchmark_method"] = benchmark_method
    out["report_rc_match_tier"] = _normalize_match_tier(match_tier)
    out["report_rc_match_tier_group"] = _tier_group(match_tier)
    out["is_selected_for_benchmark"] = out.index.isin(list(selected_index))
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
        "report_rc_match_tier",
        "report_rc_match_tier_group",
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


def match_expectations_to_events(
    events_df: pd.DataFrame,
    report_rc_df: pd.DataFrame,
    config: ProjectConfig,
    selected_tier: str = "strict_same_quarter",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_tier = _normalize_match_tier(selected_tier)
    if events_df.empty:
        return events_df.copy(), pd.DataFrame(), pd.DataFrame()

    default_method = _normalize_consensus_method(config.consensus_method)
    if report_rc_df.empty:
        out = events_df.copy()
        defaults = _default_match_values(config=config, method=default_method, tier=selected_tier)
        for col, value in defaults.items():
            out[col] = value
        audit_rows = []
        for _, row in events_df.iterrows():
            for tier in MATCH_TIERS:
                audit_rows.append({"event_id": row.get("event_id"), **_default_match_values(config, default_method, tier)})
        return out, pd.DataFrame(audit_rows), pd.DataFrame()

    report_groups = _prepare_report_groups(report_rc_df)
    matched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []

    for _, event in events_df.iterrows():
        event_date = pd.to_datetime(event.get("event_trade_date"), errors="coerce")
        period_end = pd.to_datetime(event.get("period_end"), errors="coerce")
        group = report_groups.get(event.get("ts_code"))
        tier_matches: dict[str, dict[str, object]] = {}

        for tier in MATCH_TIERS:
            method = _method_for_tier(default_method, tier)
            if group is None or group.empty:
                candidates = pd.DataFrame(columns=report_rc_df.columns)
            else:
                candidates = _candidate_universe_for_tier(event=event, group=group, config=config, tier=tier)

            if candidates.empty:
                selected = candidates.copy()
                match = _aggregate_selected_candidates(
                    selected=selected,
                    event_date=event_date,
                    config=config,
                    method=method,
                    tier=tier,
                )
            else:
                candidates = candidates.copy()
                candidates["lag_days"] = (event_date - candidates["report_date"]).dt.days
                selected = _select_consensus_candidates(candidates, method)
                match = _aggregate_selected_candidates(
                    selected=selected,
                    event_date=event_date,
                    config=config,
                    method=method,
                    tier=tier,
                )
                benchmark_field = _consensus_column(config)
                match["candidate_report_count_total"] = int(candidates[benchmark_field].notna().sum()) if benchmark_field in candidates.columns else 0
                match["candidate_broker_count_total"] = int(candidates.loc[candidates[benchmark_field].notna(), "analyst_entity"].nunique()) if benchmark_field in candidates.columns else 0
                candidate_frames.append(_candidate_rows_for_event(event, candidates, selected.index, match["benchmark_method"], tier))

            tier_matches[tier] = match
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
        row.update(tier_matches[selected_tier])
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
