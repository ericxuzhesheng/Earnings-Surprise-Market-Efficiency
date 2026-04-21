from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ProjectConfig


def _consensus_column(config: ProjectConfig) -> str:
    mapping = {
        "np": "np",
        "eps": "eps",
        "target_price": "target_price_mid",
        "np_first": "np",
    }
    return mapping.get(config.consensus_value_field, "np")


def build_expectation_panel(report_rc_df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    if report_rc_df.empty:
        return pd.DataFrame()
    df = report_rc_df.copy()
    col = _consensus_column(config)
    if col not in df.columns:
        df[col] = np.nan
    group_cols = ["ts_code", "period_end", "report_date"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            consensus_np=("np", "median"),
            consensus_eps=("eps", "median"),
            consensus_pe=("pe", "median"),
            consensus_target_price=("target_price_mid", "median"),
            consensus_rating=("rating", lambda s: s.dropna().iloc[-1] if not s.dropna().empty else np.nan),
            report_count=(col, lambda s: int(s.notna().sum())),
            broker_count=("org_name", lambda s: int(s.dropna().nunique())),
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


def match_expectations_to_events(
    events_df: pd.DataFrame,
    report_rc_df: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events_df.empty:
        return events_df.copy(), pd.DataFrame()
    if report_rc_df.empty:
        out = events_df.copy()
        out["expectation_source"] = "missing_report_rc"
        out["matched_report_date"] = pd.NaT
        out["matched_report_count"] = 0
        out["matched_broker_count"] = 0
        out["report_rc_match_quality"] = "missing"
        out["expected_np"] = np.nan
        out["expected_eps"] = np.nan
        out["expected_target_price"] = np.nan
        out["expected_rating"] = np.nan
        out["text_over_expectation_fraction"] = np.nan
        return out, pd.DataFrame()

    panel = build_expectation_panel(report_rc_df, config)
    matched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    freshness = pd.Timedelta(days=config.report_freshness_days)

    for _, event in events_df.iterrows():
        event_date = pd.to_datetime(event["event_trade_date"])
        period_end = pd.to_datetime(event["period_end"])
        candidates = panel[
            (panel["ts_code"] == event["ts_code"])
            & (panel["period_end"] == period_end)
            & (panel["report_date"] < event_date)
            & (panel["report_date"] >= event_date - freshness)
        ].copy()

        if candidates.empty:
            match = {
                "expectation_source": "missing_report_rc",
                "matched_report_date": pd.NaT,
                "matched_report_count": 0,
                "matched_broker_count": 0,
                "report_rc_match_quality": "missing",
                "expected_np": np.nan,
                "expected_eps": np.nan,
                "expected_target_price": np.nan,
                "expected_rating": np.nan,
                "text_over_expectation_fraction": np.nan,
            }
        else:
            if config.consensus_method == "latest":
                selected = candidates.sort_values("report_date").tail(1)
            elif config.consensus_method == "mean":
                selected = candidates.copy()
            else:
                selected = candidates.copy()

            min_count_ok = candidates["report_count"].max() >= config.min_valid_report_count
            if config.consensus_method == "latest":
                expected_np = selected["consensus_np"].iloc[-1]
                expected_eps = selected["consensus_eps"].iloc[-1]
                expected_target_price = selected["consensus_target_price"].iloc[-1]
                expected_rating = selected["consensus_rating"].iloc[-1]
                matched_report_date = selected["report_date"].iloc[-1]
                matched_report_count = int(selected["report_count"].iloc[-1])
                matched_broker_count = int(selected["broker_count"].iloc[-1])
                text_over_expectation_fraction = selected["fraction_title_over_expectation"].iloc[-1]
            elif config.consensus_method == "mean":
                expected_np = selected["consensus_np"].mean()
                expected_eps = selected["consensus_eps"].mean()
                expected_target_price = selected["consensus_target_price"].mean()
                expected_rating = selected["consensus_rating"].dropna().iloc[-1] if not selected["consensus_rating"].dropna().empty else np.nan
                matched_report_date = selected["report_date"].max()
                matched_report_count = int(selected["report_count"].sum())
                matched_broker_count = int(selected["broker_count"].max())
                text_over_expectation_fraction = selected["fraction_title_over_expectation"].mean()
            else:
                expected_np = selected["consensus_np"].median()
                expected_eps = selected["consensus_eps"].median()
                expected_target_price = selected["consensus_target_price"].median()
                expected_rating = selected["consensus_rating"].dropna().iloc[-1] if not selected["consensus_rating"].dropna().empty else np.nan
                matched_report_date = selected["report_date"].max()
                matched_report_count = int(selected["report_count"].sum())
                matched_broker_count = int(selected["broker_count"].max())
                text_over_expectation_fraction = selected["fraction_title_over_expectation"].median()

            match = {
                "expectation_source": "report_rc" if min_count_ok else "report_rc_below_min_count",
                "matched_report_date": matched_report_date,
                "matched_report_count": matched_report_count,
                "matched_broker_count": matched_broker_count,
                "report_rc_match_quality": "strict" if min_count_ok else "weak_count",
                "expected_np": expected_np,
                "expected_eps": expected_eps,
                "expected_target_price": expected_target_price,
                "expected_rating": expected_rating,
                "text_over_expectation_fraction": text_over_expectation_fraction,
            }

        row = event.to_dict()
        row.update(match)
        matched_rows.append(row)
        audit_rows.append(
            {
                "event_id": event.get("event_id"),
                "ts_code": event["ts_code"],
                "event_type": event["event_type"],
                "event_trade_date": event_date,
                "period_end": period_end,
                **match,
            }
        )

    return pd.DataFrame(matched_rows), pd.DataFrame(audit_rows)


def build_sell_side_revision_panel(report_rc_df: pd.DataFrame) -> pd.DataFrame:
    if report_rc_df.empty:
        return pd.DataFrame()
    df = report_rc_df.copy()
    df = df.sort_values(["ts_code", "period_end", "org_name", "report_date"]).reset_index(drop=True)
    for col in ["np", "eps", "target_price_mid"]:
        if col in df.columns:
            df[f"prev_{col}"] = df.groupby(["ts_code", "period_end", "org_name"])[col].shift(1)
            df[f"delta_{col}"] = df[col] - df[f"prev_{col}"]
    rev = (
        df.groupby(["ts_code", "period_end", "report_date"], as_index=False)
        .agg(
            revision_magnitude_np=("delta_np", "median"),
            revision_magnitude_eps=("delta_eps", "median"),
            target_price_change=("delta_target_price_mid", "median"),
            upward_revision_count=("delta_np", lambda s: int((s > 0).sum())),
            downward_revision_count=("delta_np", lambda s: int((s < 0).sum())),
            analyst_count=("org_name", lambda s: int(s.dropna().nunique())),
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
