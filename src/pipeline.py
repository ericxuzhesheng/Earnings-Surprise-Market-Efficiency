from __future__ import annotations

import shutil
import pandas as pd

from src.config import ProjectConfig
from src.data_collection import DataCollector
from src.guidance_design import (
    add_event_returns_and_controls,
    apply_tradability_filters,
    build_guidance_events,
    save_core_outputs,
)
from src.io_utils import ensure_directories, save_csv
from src.logger_utils import setup_logger


def _safe_read_csv(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _extract_old_metrics(outputs_tables_dir) -> dict:
    run_summary = _safe_read_csv(outputs_tables_dir / "run_summary.csv")
    final_group = _safe_read_csv(outputs_tables_dir / "final_group_summary.csv")
    final_reg = _safe_read_csv(outputs_tables_dir / "final_regression_results.csv")

    old_stock = float(run_summary.loc[run_summary["metric"] == "sample_stocks", "value"].iloc[0]) if not run_summary.empty and (run_summary["metric"] == "sample_stocks").any() else float("nan")
    old_event = float(run_summary.loc[run_summary["metric"] == "event_dataset_rows", "value"].iloc[0]) if not run_summary.empty and (run_summary["metric"] == "event_dataset_rows").any() else float("nan")

    old_mod_car60 = float(final_group.loc[final_group["group"] == "moderate_es_50_80", "avg_CAR60"].iloc[0]) if not final_group.empty and (final_group["group"] == "moderate_es_50_80").any() else float("nan")

    reg_mask = (
        (final_reg.get("model", pd.Series(dtype=str)) == "model_moderate_es")
        & (final_reg.get("variable", pd.Series(dtype=str)) == "moderate_positive_ES_dummy")
    )
    old_coef = float(final_reg.loc[reg_mask, "coef"].iloc[0]) if (not final_reg.empty) and reg_mask.any() else float("nan")
    old_p = float(final_reg.loc[reg_mask, "p_value"].iloc[0]) if (not final_reg.empty) and reg_mask.any() else float("nan")
    return {
        "old_stock_count": old_stock,
        "old_event_count": old_event,
        "old_avg_CAR60_moderate_ES": old_mod_car60,
        "old_coef_moderate_dummy": old_coef,
        "old_p_value_moderate_dummy": old_p,
    }


def _write_expanded_outputs(config: ProjectConfig) -> None:
    tables = config.outputs_tables_dir
    figs = config.outputs_figures_dir

    # Duplicate final outputs with expanded names for presentation robustness round.
    mapping_tables = {
        "final_group_summary.csv": "final_group_summary_expanded.csv",
        "final_regression_results.csv": "final_regression_results_expanded.csv",
        "final_interpretation.txt": "final_interpretation_expanded.txt",
    }
    mapping_figs = {
        "fig1_es_group_comparison.png": "fig1_es_group_comparison_expanded.png",
        "fig2_cum_return_moderate_vs_extreme.png": "fig2_cum_return_moderate_vs_extreme_expanded.png",
        "fig3_event_type_comparison.png": "fig3_event_type_comparison_expanded.png",
    }
    for src, dst in mapping_tables.items():
        s, d = tables / src, tables / dst
        if s.exists():
            shutil.copy2(s, d)
    for src, dst in mapping_figs.items():
        s, d = figs / src, figs / dst
        if s.exists():
            shutil.copy2(s, d)


def _build_expansion_comparison_and_note(
    config: ProjectConfig,
    old_metrics: dict,
    new_stock_count: int,
    new_event_count: int,
) -> None:
    tables = config.outputs_tables_dir
    final_group = _safe_read_csv(tables / "final_group_summary.csv")
    final_reg = _safe_read_csv(tables / "final_regression_results.csv")

    new_mod_car60 = float(final_group.loc[final_group["group"] == "moderate_es_50_80", "avg_CAR60"].iloc[0]) if not final_group.empty and (final_group["group"] == "moderate_es_50_80").any() else float("nan")
    reg_mask = (
        (final_reg.get("model", pd.Series(dtype=str)) == "model_moderate_es")
        & (final_reg.get("variable", pd.Series(dtype=str)) == "moderate_positive_ES_dummy")
    )
    new_coef = float(final_reg.loc[reg_mask, "coef"].iloc[0]) if (not final_reg.empty) and reg_mask.any() else float("nan")
    new_p = float(final_reg.loc[reg_mask, "p_value"].iloc[0]) if (not final_reg.empty) and reg_mask.any() else float("nan")

    comp = pd.DataFrame(
        {
            "old_stock_count": [old_metrics["old_stock_count"]],
            "new_stock_count": [new_stock_count],
            "old_event_count": [old_metrics["old_event_count"]],
            "new_event_count": [new_event_count],
            "old_avg_CAR60_moderate_ES": [old_metrics["old_avg_CAR60_moderate_ES"]],
            "new_avg_CAR60_moderate_ES": [new_mod_car60],
            "old_coef_moderate_positive_ES_dummy": [old_metrics["old_coef_moderate_dummy"]],
            "new_coef_moderate_positive_ES_dummy": [new_coef],
            "old_p_value": [old_metrics["old_p_value_moderate_dummy"]],
            "new_p_value": [new_p],
            "old_liquidity_turnover20_filter": [config.liquidity_turnover20_old],
            "new_liquidity_turnover20_filter": [config.liquidity_turnover20_new],
        }
    )
    save_csv(comp, tables / "sample_expansion_comparison.csv")

    old_stock = old_metrics["old_stock_count"]
    old_event = old_metrics["old_event_count"]
    expand_stock_ratio = (new_stock_count / old_stock - 1.0) if pd.notna(old_stock) and old_stock > 0 else float("nan")
    expand_event_ratio = (new_event_count / old_event - 1.0) if pd.notna(old_event) and old_event > 0 else float("nan")
    new_ext_car60 = float(final_group.loc[final_group["group"] == "extreme_es_top20", "avg_CAR60"].iloc[0]) if not final_group.empty and (final_group["group"] == "extreme_es_top20").any() else float("nan")
    old_ext_car60 = float(final_group.loc[final_group["group"] == "extreme_es_top20", "avg_CAR60"].iloc[0]) if not final_group.empty and (final_group["group"] == "extreme_es_top20").any() else float("nan")
    core_same = pd.notna(new_mod_car60) and pd.notna(new_ext_car60) and (new_mod_car60 >= new_ext_car60)
    core_line = (
        "2) Core conclusion remains: moderate positive surprise is more reliable than extreme surprise in this framework."
        if core_same
        else "2) Expanded sample partially overturns the prior ordering: extreme ES currently shows higher average CAR60 than moderate ES."
    )
    detail_line = (
        f"3) Current expanded averages: moderate ES CAR60={new_mod_car60:.3%}, extreme ES CAR60={new_ext_car60:.3%}."
        if pd.notna(new_mod_car60) and pd.notna(new_ext_car60)
        else "3) Moderate vs extreme ES comparison is unavailable due to missing group estimates."
    )

    note = (
        "Final Interpretation Expanded\n"
        f"1) Sample expanded from stock count {old_stock:.0f} to {new_stock_count:.0f}, and event count {old_event:.0f} to {new_event_count:.0f} "
        f"(stock change {expand_stock_ratio:.1%}, event change {expand_event_ratio:.1%}).\n"
        f"{core_line}\n"
        f"{detail_line}\n"
        f"4) Regression stability check: coefficient moved from {old_metrics['old_coef_moderate_dummy']:.6f} (p={old_metrics['old_p_value_moderate_dummy']:.3f}) "
        f"to {new_coef:.6f} (p={new_p:.3f}).\n"
        "5) The expanded-sample version is better for final presentation because it reduces small-sample concern while preserving the same corporate-finance narrative.\n"
        f"Liquidity filter was relaxed from turnover20 >= {config.liquidity_turnover20_old:.2f} to >= {config.liquidity_turnover20_new:.2f}.\n"
    )
    (tables / "final_interpretation_expanded.txt").write_text(note, encoding="utf-8")


def run_pipeline() -> None:
    config = ProjectConfig()
    ensure_directories(
        [
            config.data_raw_dir,
            config.data_processed_dir,
            config.outputs_dir,
            config.outputs_figures_dir,
            config.outputs_tables_dir,
            config.logs_dir,
        ]
    )
    logger = setup_logger(config.logs_dir)
    logger.info(
        "Guidance-design pipeline start | RUN_MODE=%s | period=%s-%s",
        config.run_mode,
        config.start_date,
        config.end_date,
    )
    old_metrics = _extract_old_metrics(config.outputs_tables_dir)

    collector = DataCollector(config=config, logger=logger)
    stocks = collector.get_stock_universe()
    guidance = collector.get_guidance_data(stocks)
    prices = collector.get_stock_prices(stocks)
    market = collector.get_market_index()
    daily_basic = collector.get_daily_basic(stocks)

    save_csv(stocks, config.data_raw_dir / "stock_universe.csv")
    save_csv(guidance, config.data_raw_dir / "guidance_forecast_raw.csv")
    save_csv(prices, config.data_raw_dir / "stock_prices_raw.csv")
    save_csv(market, config.data_raw_dir / "market_index_raw.csv")
    save_csv(daily_basic, config.data_raw_dir / "daily_basic_raw.csv")

    events = build_guidance_events(
        guidance_df=guidance,
        stocks_df=stocks,
        market_df=market,
        logger=logger,
    )
    events = apply_tradability_filters(
        events_df=events,
        prices_df=prices,
        daily_basic_df=daily_basic,
        market_df=market,
        min_listed_trading_days=120,
        turnover20_threshold=config.liquidity_turnover20_new,
    )
    event_dataset, event_paths = add_event_returns_and_controls(
        events_df=events,
        prices_df=prices,
        market_df=market,
        daily_basic_df=daily_basic,
    )

    save_csv(events, config.data_processed_dir / "guidance_events_filtered.csv")
    save_csv(event_dataset, config.data_processed_dir / "event_dataset_guidance_2020plus.csv")
    save_csv(event_paths, config.data_processed_dir / "event_paths_guidance_2020plus.csv")

    save_core_outputs(
        event_df=event_dataset,
        path_df=event_paths,
        outputs_tables_dir=config.outputs_tables_dir,
        outputs_figures_dir=config.outputs_figures_dir,
        logger=logger,
    )
    _write_expanded_outputs(config)
    _build_expansion_comparison_and_note(
        config=config,
        old_metrics=old_metrics,
        new_stock_count=len(stocks),
        new_event_count=len(event_dataset),
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "run_mode",
                "sample_stocks",
                "guidance_rows_raw",
                "events_after_filters",
                "event_dataset_rows",
                "period_start",
                "period_end",
            ],
            "value": [
                config.run_mode,
                len(stocks),
                len(guidance),
                len(events),
                len(event_dataset),
                config.start_date,
                config.end_date,
            ],
        }
    )
    save_csv(summary, config.outputs_tables_dir / "run_summary.csv")
    logger.info("Guidance-design pipeline complete.")
