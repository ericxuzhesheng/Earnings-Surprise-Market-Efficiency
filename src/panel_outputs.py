from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import save_csv, save_text
from src.tushare_event_design import (
    build_ablation_summary_for_run,
    build_failure_analysis,
    build_recommendation_note,
    choose_strongest_spec,
    recommendation_from_diagnostics,
)


SIGNAL_COLUMN_MAP = {
    "raw": "main_surprise_raw",
    "pct": "main_surprise_pct",
    "std": "main_surprise_std",
}


def _prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["event_trade_date"] = pd.to_datetime(out["event_trade_date"], errors="coerce")
    out["year"] = out["event_trade_date"].dt.year
    out["year_quarter"] = out["event_trade_date"].dt.to_period("Q").astype(str)
    out["industry"] = out["industry"].fillna("unknown")
    out["event_type"] = out["event_type"].fillna("unknown")
    out["report_rc_match_tier"] = out.get("report_rc_match_tier", pd.Series(index=out.index, dtype=object)).fillna("strict_same_quarter")
    out["report_rc_match_tier_group"] = out.get("report_rc_match_tier_group", pd.Series(index=out.index, dtype=object)).fillna(
        np.where(out["report_rc_match_tier"].eq("strict_same_quarter"), "strict", "relaxed")
    )
    return out


def _missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_share"])
    missing_cols = [
        "main_surprise_raw",
        "main_surprise_pct",
        "main_surprise_std",
        "expected_np",
        "expected_eps",
        "expected_value_primary",
        "matched_report_count",
        "matched_broker_count",
        "benchmark_lag_days",
        "turnover20",
        "beta",
        "total_mv",
        "pb",
        "ps_ttm",
    ] + [c for c in df.columns if c.startswith("CAR")]
    missing_cols = list(dict.fromkeys(missing_cols))
    return pd.DataFrame(
        {
            "column": missing_cols,
            "missing_count": [int(df[c].isna().sum()) if c in df.columns else len(df) for c in missing_cols],
            "missing_share": [float(df[c].isna().mean()) if c in df.columns else 1.0 for c in missing_cols],
        }
    )


def _coverage_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    coverage = (
        df.groupby("year", as_index=False)
        .agg(
            event_count=("event_id", "count"),
            strict_report_rc_count=("report_rc_match_quality", lambda s: int((s == "strict").sum())),
            strict_tier_count=("report_rc_match_tier", lambda s: int((s == "strict_same_quarter").sum())),
            usable_raw_signal_count=("main_surprise_raw", lambda s: int(s.notna().sum())),
            usable_pct_signal_count=("main_surprise_pct", lambda s: int(s.notna().sum())),
            usable_std_signal_count=("main_surprise_std", lambda s: int(s.notna().sum())),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    for col in [
        "strict_report_rc_count",
        "strict_tier_count",
        "usable_raw_signal_count",
        "usable_pct_signal_count",
        "usable_std_signal_count",
    ]:
        coverage[col.replace("_count", "_share")] = coverage[col] / coverage["event_count"].replace(0, np.nan)
    return coverage


def _coverage_by_match_tier(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["report_rc_match_tier", "report_rc_match_tier_group"], as_index=False)
        .agg(
            event_count=("event_id", "count"),
            strict_report_rc_count=("report_rc_match_quality", lambda s: int((s == "strict").sum())),
            usable_raw_signal_count=("main_surprise_raw", lambda s: int(s.notna().sum())),
            usable_pct_signal_count=("main_surprise_pct", lambda s: int(s.notna().sum())),
            usable_std_signal_count=("main_surprise_std", lambda s: int(s.notna().sum())),
            median_broker_count=("matched_broker_count", "median"),
            median_lag_days=("benchmark_lag_days", "median"),
        )
        .sort_values(["report_rc_match_tier_group", "report_rc_match_tier"])
        .reset_index(drop=True)
    )
    out["strict_report_rc_share"] = out["strict_report_rc_count"] / out["event_count"].replace(0, np.nan)
    out["usable_raw_signal_share"] = out["usable_raw_signal_count"] / out["event_count"].replace(0, np.nan)
    out["usable_pct_signal_share"] = out["usable_pct_signal_count"] / out["event_count"].replace(0, np.nan)
    out["usable_std_signal_share"] = out["usable_std_signal_count"] / out["event_count"].replace(0, np.nan)
    return out


def _best_by_dimension(results_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if results_df is None or results_df.empty or group_col not in results_df.columns:
        return pd.DataFrame()
    scored = results_df.copy()
    scored = scored[scored["coef"].notna() & scored["p_value"].notna()].copy()
    if scored.empty:
        return pd.DataFrame()
    scored = scored.sort_values([group_col, "p_value", "regression_nobs"], ascending=[True, True, False])
    return scored.groupby(group_col, as_index=False).head(1).reset_index(drop=True)


def _headline_signal_table(reg_df: pd.DataFrame) -> pd.DataFrame:
    if reg_df.empty:
        return pd.DataFrame()
    signal_rows = reg_df[(reg_df["model"] == "pooled_fe") & (reg_df["variable"].isin(SIGNAL_COLUMN_MAP.values()))].copy()
    if signal_rows.empty:
        return pd.DataFrame()
    signal_rows["signal_scale"] = signal_rows["variable"].map({v: k for k, v in SIGNAL_COLUMN_MAP.items()})
    return signal_rows.sort_values(["dependent_var", "signal_scale"]).reset_index(drop=True)


def _main_display_spec(headline_signal_df: pd.DataFrame, strongest_spec_df: pd.DataFrame) -> dict[str, object]:
    if strongest_spec_df is not None and not strongest_spec_df.empty:
        row = strongest_spec_df.iloc[0]
        return {
            "headline_signal_scale": row.get("signal_scale", "raw"),
            "headline_window": row.get("car_window", "CAR5"),
            "headline_match_tier": row.get("match_tier", "strict_same_quarter"),
            "headline_coef": row.get("coef", np.nan),
            "headline_p_value": row.get("p_value", np.nan),
        }
    if headline_signal_df.empty:
        return {
            "headline_signal_scale": "raw",
            "headline_window": "CAR5",
            "headline_match_tier": "strict_same_quarter",
            "headline_coef": np.nan,
            "headline_p_value": np.nan,
        }
    scored = headline_signal_df.copy()
    scored["signal_rank"] = scored["signal_scale"].map({"raw": 0, "pct": 1, "std": 2}).fillna(9)
    scored = scored.sort_values(["p_value", "signal_rank", "n_obs"], ascending=[True, True, False])
    row = scored.iloc[0]
    return {
        "headline_signal_scale": row.get("signal_scale", "raw"),
        "headline_window": row.get("dependent_var", "CAR5"),
        "headline_match_tier": "strict_same_quarter",
        "headline_coef": row.get("coef", np.nan),
        "headline_p_value": row.get("p_value", np.nan),
    }


def save_tushare_outputs(
    event_df: pd.DataFrame,
    path_df: pd.DataFrame,
    outputs_tables_dir: Path,
    outputs_audit_dir: Path,
    scenario_name: str = "tushare_first",
    diagnostic_panel_df: pd.DataFrame | None = None,
    supplementary_event_df: pd.DataFrame | None = None,
    supplementary_path_df: pd.DataFrame | None = None,
    filter_funnel_df: pd.DataFrame | None = None,
    benchmark_quality_df: pd.DataFrame | None = None,
    timing_alignment_df: pd.DataFrame | None = None,
    expectation_coverage_df: pd.DataFrame | None = None,
    event_signal_df: pd.DataFrame | None = None,
    ablation_catalog_df: pd.DataFrame | None = None,
    ablation_results_df: pd.DataFrame | None = None,
    window_availability_df: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    outputs_audit_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = outputs_tables_dir / f"event_dataset_{scenario_name}.csv"
    path_path = outputs_tables_dir / f"event_paths_{scenario_name}.csv"
    supplementary_dataset_path = outputs_tables_dir / f"event_dataset_{scenario_name}_all_events.csv"
    supplementary_path_path = outputs_tables_dir / f"event_paths_{scenario_name}_all_events.csv"
    reg_path = outputs_tables_dir / f"regression_results_{scenario_name}.csv"
    supplementary_reg_path = outputs_tables_dir / f"regression_results_{scenario_name}_all_events.csv"
    event_type_path = outputs_tables_dir / f"event_counts_by_type_{scenario_name}.csv"
    supplementary_event_type_path = outputs_tables_dir / f"event_counts_by_type_{scenario_name}_all_events.csv"
    year_path = outputs_tables_dir / f"event_counts_by_year_{scenario_name}.csv"
    supplementary_year_path = outputs_tables_dir / f"event_counts_by_year_{scenario_name}_all_events.csv"
    coverage_path = outputs_audit_dir / f"signal_coverage_by_year_{scenario_name}.csv"
    supplementary_coverage_path = outputs_audit_dir / f"signal_coverage_by_year_{scenario_name}_all_events.csv"
    coverage_by_tier_path = outputs_audit_dir / f"coverage_by_match_tier_{scenario_name}.csv"
    missing_path = outputs_audit_dir / f"missingness_summary_{scenario_name}.csv"
    supplementary_missing_path = outputs_audit_dir / f"missingness_summary_{scenario_name}_all_events.csv"
    audit_note_path = outputs_audit_dir / f"audit_note_{scenario_name}.txt"
    supplementary_audit_note_path = outputs_audit_dir / f"audit_note_{scenario_name}_all_events.txt"
    recommendation_path = outputs_audit_dir / f"final_recommendation_{scenario_name}.txt"
    event_signal_path = outputs_audit_dir / f"event_type_signal_summary_{scenario_name}.csv"
    window_path = outputs_audit_dir / f"window_availability_{scenario_name}.csv"
    expectation_coverage_path = outputs_audit_dir / f"expectation_coverage_{scenario_name}.csv"
    benchmark_path = outputs_audit_dir / f"benchmark_quality_{scenario_name}.csv"
    timing_path = outputs_audit_dir / f"timing_alignment_{scenario_name}.csv"
    funnel_path = outputs_audit_dir / f"coverage_funnel_{scenario_name}.csv"
    failure_path = outputs_audit_dir / f"failure_analysis_{scenario_name}.csv"
    ablation_catalog_path = outputs_tables_dir / f"ablation_catalog_{scenario_name}.csv"
    ablation_results_path = outputs_tables_dir / f"ablation_results_{scenario_name}.csv"
    strongest_spec_path = outputs_tables_dir / f"strongest_surviving_spec_{scenario_name}.csv"
    headline_compare_path = outputs_tables_dir / f"headline_signal_comparison_{scenario_name}.csv"
    match_regression_path = outputs_tables_dir / f"regression_results_by_match_tier_{scenario_name}.csv"
    universe_compare_path = outputs_tables_dir / f"diagnostic_compare_event_universe_{scenario_name}.csv"
    signal_compare_path = outputs_tables_dir / f"diagnostic_compare_signal_scale_{scenario_name}.csv"
    window_compare_path = outputs_tables_dir / f"diagnostic_compare_car_window_{scenario_name}.csv"
    match_compare_path = outputs_tables_dir / f"diagnostic_compare_match_tier_{scenario_name}.csv"
    analyst_compare_path = outputs_tables_dir / f"diagnostic_compare_analyst_threshold_{scenario_name}.csv"
    strict_relaxed_compare_path = outputs_tables_dir / f"diagnostic_compare_strict_relaxed_{scenario_name}.csv"
    narrative_note_path = outputs_tables_dir / f"project_narrative_update_note_{scenario_name}.txt"
    focused_note_path = outputs_audit_dir / f"focused_diagnostic_package_note_{scenario_name}.txt"

    df = _prepare_panel(event_df)
    supp_df = _prepare_panel(supplementary_event_df if supplementary_event_df is not None else pd.DataFrame())
    diag_df = _prepare_panel(diagnostic_panel_df if diagnostic_panel_df is not None else df)

    save_csv(df, dataset_path)
    save_csv(path_df if path_df is not None else pd.DataFrame(), path_path)
    save_csv(supp_df, supplementary_dataset_path)
    save_csv(supplementary_path_df if supplementary_path_df is not None else pd.DataFrame(), supplementary_path_path)

    empty_outputs = [
        reg_path,
        supplementary_reg_path,
        event_type_path,
        supplementary_event_type_path,
        year_path,
        supplementary_year_path,
        coverage_path,
        supplementary_coverage_path,
        coverage_by_tier_path,
        missing_path,
        supplementary_missing_path,
        benchmark_path,
        timing_path,
        expectation_coverage_path,
        event_signal_path,
        window_path,
        funnel_path,
        failure_path,
        ablation_catalog_path,
        ablation_results_path,
        strongest_spec_path,
        headline_compare_path,
        match_regression_path,
        universe_compare_path,
        signal_compare_path,
        window_compare_path,
        match_compare_path,
        analyst_compare_path,
        strict_relaxed_compare_path,
    ]
    if df.empty:
        empty = pd.DataFrame()
        for path in empty_outputs:
            save_csv(empty, path)
        save_text("No valid headline events after filters.\n", audit_note_path)
        save_text("Tushare-first diagnostic recommendation\nRecommendation: B\nNo valid headline events after filters.\n", recommendation_path)
        save_text("The focused diagnostic package is empty because the headline preannouncement-only sample has no valid rows.\n", focused_note_path)
        save_text("The repository remains a Tushare-based diagnostic baseline; this run produced no usable headline sample.\n", narrative_note_path)
        return {
            "scenario": scenario_name,
            "sample_size": 0,
            "headline_window": "CAR5",
            "headline_coef": np.nan,
            "headline_p_value": np.nan,
            "usable_signal_rows": 0,
            "strict_match_rows": 0,
            "headline_signal_scale": "raw",
            "headline_match_tier": "strict_same_quarter",
            "diagnostic_recommendation": "B",
            "strongest_spec_id": "",
            "strongest_spec_window": "",
            "strongest_spec_coef": np.nan,
            "strongest_spec_p_value": np.nan,
            "strongest_spec_nobs": 0,
        }

    event_counts_by_type = df.groupby("event_type", as_index=False).size().rename(columns={"size": "event_count"})
    supplementary_counts_by_type = supp_df.groupby("event_type", as_index=False).size().rename(columns={"size": "event_count"}) if not supp_df.empty else pd.DataFrame()
    event_counts_by_year = df.groupby("year", as_index=False).size().rename(columns={"size": "event_count"})
    supplementary_counts_by_year = supp_df.groupby("year", as_index=False).size().rename(columns={"size": "event_count"}) if not supp_df.empty else pd.DataFrame()
    coverage = _coverage_by_year(df)
    supplementary_coverage = _coverage_by_year(supp_df)
    coverage_by_tier = _coverage_by_match_tier(diag_df)
    missingness = _missingness_summary(df)
    supplementary_missingness = _missingness_summary(supp_df)

    reg = run_tushare_regressions(df)
    supplementary_reg = run_tushare_regressions(supp_df)
    headline_signal_comparison = _headline_signal_table(reg)
    strongest_spec = choose_strongest_spec(ablation_results_df) if ablation_results_df is not None else pd.DataFrame()
    ablation_summary = build_ablation_summary_for_run(ablation_results_df) if ablation_results_df is not None else {
        "strongest_spec_id": "",
        "strongest_spec_window": "",
        "strongest_spec_coef": np.nan,
        "strongest_spec_p_value": np.nan,
        "strongest_spec_nobs": 0,
    }
    main_display = _main_display_spec(headline_signal_comparison, strongest_spec)

    failure_analysis = build_failure_analysis(
        event_panel_df=diag_df,
        spec_results_df=ablation_results_df if ablation_results_df is not None else pd.DataFrame(),
        filter_funnel_df=filter_funnel_df if filter_funnel_df is not None else pd.DataFrame(),
        benchmark_quality_df=benchmark_quality_df if benchmark_quality_df is not None else pd.DataFrame(),
    )
    recommendation_code, recommendation_text = recommendation_from_diagnostics(
        strongest_spec_df=strongest_spec,
        failure_df=failure_analysis,
    )
    recommendation_note = build_recommendation_note(recommendation_code, recommendation_text, strongest_spec)

    match_regression = pd.DataFrame()
    if ablation_results_df is not None and not ablation_results_df.empty:
        match_regression = ablation_results_df.sort_values(["match_tier", "signal_scale", "car_window", "p_value", "regression_nobs"]).reset_index(drop=True)

    universe_compare = _best_by_dimension(ablation_results_df if ablation_results_df is not None else pd.DataFrame(), "event_universe")
    signal_compare = _best_by_dimension(ablation_results_df if ablation_results_df is not None else pd.DataFrame(), "signal_scale")
    window_compare = _best_by_dimension(ablation_results_df if ablation_results_df is not None else pd.DataFrame(), "car_window")
    match_compare = _best_by_dimension(ablation_results_df if ablation_results_df is not None else pd.DataFrame(), "match_tier")
    analyst_compare = _best_by_dimension(ablation_results_df if ablation_results_df is not None else pd.DataFrame(), "analyst_min")
    strict_relaxed_compare = _best_by_dimension(ablation_results_df if ablation_results_df is not None else pd.DataFrame(), "match_tier_group")

    save_csv(event_counts_by_type, event_type_path)
    save_csv(supplementary_counts_by_type, supplementary_event_type_path)
    save_csv(event_counts_by_year, year_path)
    save_csv(supplementary_counts_by_year, supplementary_year_path)
    save_csv(coverage, coverage_path)
    save_csv(supplementary_coverage, supplementary_coverage_path)
    save_csv(coverage_by_tier, coverage_by_tier_path)
    save_csv(missingness, missing_path)
    save_csv(supplementary_missingness, supplementary_missing_path)
    save_csv(reg, reg_path)
    save_csv(supplementary_reg, supplementary_reg_path)
    save_csv(filter_funnel_df if filter_funnel_df is not None else pd.DataFrame(), funnel_path)
    save_csv(benchmark_quality_df if benchmark_quality_df is not None else pd.DataFrame(), benchmark_path)
    save_csv(timing_alignment_df if timing_alignment_df is not None else pd.DataFrame(), timing_path)
    save_csv(expectation_coverage_df if expectation_coverage_df is not None else pd.DataFrame(), expectation_coverage_path)
    save_csv(event_signal_df if event_signal_df is not None else pd.DataFrame(), event_signal_path)
    save_csv(window_availability_df if window_availability_df is not None else pd.DataFrame(), window_path)
    save_csv(ablation_catalog_df if ablation_catalog_df is not None else pd.DataFrame(), ablation_catalog_path)
    save_csv(ablation_results_df if ablation_results_df is not None else pd.DataFrame(), ablation_results_path)
    save_csv(strongest_spec, strongest_spec_path)
    save_csv(failure_analysis, failure_path)
    save_csv(headline_signal_comparison, headline_compare_path)
    save_csv(match_regression, match_regression_path)
    save_csv(universe_compare, universe_compare_path)
    save_csv(signal_compare, signal_compare_path)
    save_csv(window_compare, window_compare_path)
    save_csv(match_compare, match_compare_path)
    save_csv(analyst_compare, analyst_compare_path)
    save_csv(strict_relaxed_compare, strict_relaxed_compare_path)
    save_text(recommendation_note, recommendation_path)

    audit_lines = [
        f"Tushare-first audit note ({scenario_name})",
        f"Headline sample size: {len(df)}",
        f"Supplementary all-event sample size: {len(supp_df)}",
        f"Diagnostic base panel size: {len(diag_df)}",
        f"Headline event types: {', '.join(sorted(df['event_type'].dropna().astype(str).unique()))}",
        f"Strict same-quarter tier rows in headline sample: {(df['report_rc_match_tier'] == 'strict_same_quarter').sum()} / {len(df)}",
        f"Usable raw/pct/std rows: {df['main_surprise_raw'].notna().sum()} / {df['main_surprise_pct'].notna().sum()} / {df['main_surprise_std'].notna().sum()}",
        f"Main display spec: {main_display['headline_window']} on {main_display['headline_signal_scale']} surprise within the preannouncement-only strict same-quarter headline sample.",
        f"Diagnostic recommendation: {recommendation_code}",
        recommendation_text,
    ]
    save_text("\n".join(audit_lines) + "\n", audit_note_path)

    supplementary_audit_lines = [
        f"Supplementary all-event audit note ({scenario_name})",
        f"Sample size: {len(supp_df)}",
        f"Event types: {', '.join(sorted(supp_df['event_type'].dropna().astype(str).unique())) if not supp_df.empty else ''}",
        "This file is supplementary only; the headline sample is preannouncement-only.",
    ]
    save_text("\n".join(supplementary_audit_lines) + "\n", supplementary_audit_note_path)

    narrative_lines = [
        "Tushare-first narrative update",
        "The repository should be framed as a Tushare-based diagnostic baseline.",
        "The headline sample is now preannouncement_only under strict_same_quarter matching.",
        "Raw, pct, and std surprises are compared in parallel; standardization is robustness, not the only headline lens.",
        "The main research message is that expectation measurement and match quality materially affect inference.",
        "Do not claim strong headline evidence from the current Tushare-only sample.",
    ]
    save_text("\n".join(narrative_lines) + "\n", narrative_note_path)

    focused_note_lines = [
        "Focused diagnostic package",
        "Included side-by-side tables:",
        f"- preannouncement_only vs all_event_types: {universe_compare_path.name}",
        f"- raw vs pct vs std: {signal_compare_path.name}",
        f"- CAR3 / CAR5 / CAR10 / CAR20: {window_compare_path.name}",
        f"- strict vs relaxed matching tiers: {strict_relaxed_compare_path.name}",
        f"- explicit match tiers: {match_compare_path.name}",
        f"- minimum analyst coverage thresholds: {analyst_compare_path.name}",
    ]
    save_text("\n".join(focused_note_lines) + "\n", focused_note_path)

    return {
        "scenario": scenario_name,
        "sample_size": int(len(df)),
        "headline_window": str(main_display["headline_window"]),
        "headline_coef": float(main_display["headline_coef"]) if pd.notna(main_display["headline_coef"]) else np.nan,
        "headline_p_value": float(main_display["headline_p_value"]) if pd.notna(main_display["headline_p_value"]) else np.nan,
        "usable_signal_rows": int(df["main_surprise_raw"].notna().sum()),
        "strict_match_rows": int((df["report_rc_match_quality"] == "strict").sum()),
        "headline_signal_scale": str(main_display["headline_signal_scale"]),
        "headline_match_tier": str(main_display["headline_match_tier"]),
        "diagnostic_recommendation": recommendation_code,
        **ablation_summary,
    }


def run_tushare_regressions(df: pd.DataFrame) -> pd.DataFrame:
    df = _prepare_panel(df)
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception:
        return pd.DataFrame()

    car_windows = sorted(c for c in df.columns if c.startswith("CAR") and not c.endswith("_available"))
    controls = ["log_total_mv", "beta", "book_to_market", "turnover20", "pe_ttm", "ps_ttm"]

    for dep in car_windows:
        for signal_scale, signal_col in SIGNAL_COLUMN_MAP.items():
            regressors = [signal_col] + controls
            cols = [dep, "ts_code", "industry", "year_quarter", "event_type", "report_rc_match_tier", "report_rc_match_tier_group"] + regressors
            dd = df[[c for c in cols if c in df.columns]].copy()
            for col in [dep] + regressors:
                if col in dd.columns:
                    dd[col] = pd.to_numeric(dd[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            dd = dd.dropna(subset=[dep, signal_col, "ts_code", "industry", "year_quarter", "event_type"]).copy()
            if len(dd) < 30:
                rows.append(
                    {
                        "model": "pooled_fe",
                        "dependent_var": dep,
                        "variable": signal_col,
                        "signal_scale": signal_scale,
                        "coef": np.nan,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "n_obs": int(len(dd)),
                        "r2": np.nan,
                        "report_rc_match_tier": "strict_same_quarter",
                        "report_rc_match_tier_group": "strict",
                    }
                )
                continue
            active_regs = [r for r in regressors if r in dd.columns and dd[r].notna().sum() >= 20]
            reg_dd = dd[[dep, "ts_code", "industry", "year_quarter", "event_type"] + active_regs].dropna().copy()
            if len(reg_dd) < 30 or signal_col not in active_regs:
                rows.append(
                    {
                        "model": "pooled_fe",
                        "dependent_var": dep,
                        "variable": signal_col,
                        "signal_scale": signal_scale,
                        "coef": np.nan,
                        "t_stat": np.nan,
                        "p_value": np.nan,
                        "n_obs": int(len(reg_dd)),
                        "r2": np.nan,
                        "report_rc_match_tier": "strict_same_quarter",
                        "report_rc_match_tier_group": "strict",
                    }
                )
                continue
            formula = f"{dep} ~ {' + '.join(active_regs)} + C(industry) + C(year_quarter) + C(event_type)"
            model = smf.ols(formula=formula, data=reg_dd).fit(cov_type="cluster", cov_kwds={"groups": reg_dd["ts_code"]})
            for var in active_regs:
                rows.append(
                    {
                        "model": "pooled_fe",
                        "dependent_var": dep,
                        "variable": var,
                        "signal_scale": signal_scale,
                        "coef": model.params.get(var, np.nan),
                        "t_stat": model.tvalues.get(var, np.nan),
                        "p_value": model.pvalues.get(var, np.nan),
                        "n_obs": int(model.nobs),
                        "r2": model.rsquared,
                        "report_rc_match_tier": "strict_same_quarter",
                        "report_rc_match_tier_group": "strict",
                    }
                )
    return pd.DataFrame(rows)
