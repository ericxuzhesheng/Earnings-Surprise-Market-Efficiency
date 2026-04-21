from __future__ import annotations

import pandas as pd

from src.config import ProjectConfig
from src.data_collection import DataCollector
from src.guidance_design import (
    add_event_returns_and_controls,
    apply_tradability_filters,
    build_guidance_events,
    save_core_outputs,
)
from src.io_utils import ensure_directories, save_csv, save_text
from src.logger_utils import setup_logger


def _build_before_after_comparison(
    config: ProjectConfig,
    baseline_metrics: dict[str, float | str],
    improved_metrics: dict[str, float | str],
    baseline_events: pd.DataFrame,
    improved_events: pd.DataFrame,
    baseline_reg: pd.DataFrame,
    improved_reg: pd.DataFrame,
) -> None:
    tables = config.outputs_tables_dir

    comparison = pd.DataFrame(
        {
            "metric": [
                "sample_size",
                "primary_car",
                "moderate_group_mean",
                "extreme_group_mean",
                "key_coef",
                "key_p_value",
            ],
            "baseline": [
                baseline_metrics.get("sample_size"),
                baseline_metrics.get("primary_car"),
                baseline_metrics.get("moderate_group_mean"),
                baseline_metrics.get("extreme_group_mean"),
                baseline_metrics.get("coef"),
                baseline_metrics.get("p_value"),
            ],
            "improved": [
                improved_metrics.get("sample_size"),
                improved_metrics.get("primary_car"),
                improved_metrics.get("moderate_group_mean"),
                improved_metrics.get("extreme_group_mean"),
                improved_metrics.get("coef"),
                improved_metrics.get("p_value"),
            ],
        }
    )
    save_csv(comparison, tables / "before_after_method_comparison.csv")

    baseline_car_cols = [c for c in baseline_events.columns if c.startswith("CAR")]
    improved_car_cols = [c for c in improved_events.columns if c.startswith("CAR")]
    sample_comp = pd.DataFrame(
        {
            "scenario": ["baseline", "improved"],
            "event_rows": [len(baseline_events), len(improved_events)],
            "car_columns": [", ".join(baseline_car_cols), ", ".join(improved_car_cols)],
        }
    )
    save_csv(sample_comp, tables / "before_after_sample_sizes.csv")

    def _coef_line(metrics: dict[str, float | str], label: str) -> str:
        return f"{label}: coef={metrics.get('coef', float('nan')):.6f}, p={metrics.get('p_value', float('nan')):.3f}, N={int(metrics.get('sample_size', 0) or 0)}"

    text = "\n".join(
        [
            "Before vs After Comparison",
            "Baseline uses legacy ES_main, longer CAR window, and clustered OLS on the moderate-positive dummy.",
            "Improved uses standardized ES_std, short-window CAR, and panel regression with firm/time fixed effects.",
            _coef_line(baseline_metrics, "Baseline"),
            _coef_line(improved_metrics, "Improved"),
            f"Baseline primary window: {baseline_metrics.get('primary_car')}; Improved primary window: {improved_metrics.get('primary_car')}.",
            f"Moderate-vs-extreme comparison moved from {baseline_metrics.get('moderate_group_mean', float('nan')):.3%} vs {baseline_metrics.get('extreme_group_mean', float('nan')):.3%} to {improved_metrics.get('moderate_group_mean', float('nan')):.3%} vs {improved_metrics.get('extreme_group_mean', float('nan')):.3%}.",
            "Preferred presentation should emphasize the improved specification only if the sign and magnitude are economically interpretable and robust, not merely smaller p-values.",
        ]
    )
    save_text(text + "\n", tables / "before_after_interpretation.txt")


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

    baseline_events = apply_tradability_filters(
        events_df=events.copy(),
        prices_df=prices,
        daily_basic_df=daily_basic,
        market_df=market,
        min_listed_trading_days=120,
        turnover20_threshold=config.liquidity_turnover20_old,
    )
    improved_events = apply_tradability_filters(
        events_df=events.copy(),
        prices_df=prices,
        daily_basic_df=daily_basic,
        market_df=market,
        min_listed_trading_days=120,
        turnover20_threshold=config.liquidity_turnover20_new,
    )

    baseline_dataset, baseline_paths = add_event_returns_and_controls(
        events_df=baseline_events,
        prices_df=prices,
        market_df=market,
        daily_basic_df=daily_basic,
        event_windows=(20, 60),
    )
    improved_dataset, improved_paths = add_event_returns_and_controls(
        events_df=improved_events,
        prices_df=prices,
        market_df=market,
        daily_basic_df=daily_basic,
        event_windows=(3, 5, 20),
    )

    save_csv(events, config.data_processed_dir / "guidance_events_all.csv")
    save_csv(baseline_events, config.data_processed_dir / "guidance_events_filtered_baseline.csv")
    save_csv(improved_events, config.data_processed_dir / "guidance_events_filtered_improved.csv")
    save_csv(baseline_dataset, config.data_processed_dir / "event_dataset_guidance_baseline.csv")
    save_csv(improved_dataset, config.data_processed_dir / "event_dataset_guidance_improved.csv")
    save_csv(baseline_paths, config.data_processed_dir / "event_paths_guidance_baseline.csv")
    save_csv(improved_paths, config.data_processed_dir / "event_paths_guidance_improved.csv")

    baseline_metrics = save_core_outputs(
        event_df=baseline_dataset,
        path_df=baseline_paths,
        outputs_tables_dir=config.outputs_tables_dir,
        outputs_figures_dir=config.outputs_figures_dir,
        logger=logger,
        scenario_name="baseline",
        primary_car="CAR60",
        car_windows=(20, 60),
        signal_col="earnings_surprise",
        use_panel_regression=False,
    )
    improved_metrics = save_core_outputs(
        event_df=improved_dataset,
        path_df=improved_paths,
        outputs_tables_dir=config.outputs_tables_dir,
        outputs_figures_dir=config.outputs_figures_dir,
        logger=logger,
        scenario_name="improved",
        primary_car="CAR5",
        car_windows=(3, 5, 20),
        signal_col="ES_std",
        use_panel_regression=True,
    )

    baseline_reg = pd.read_csv(config.outputs_tables_dir / "final_regression_results_baseline.csv")
    improved_reg = pd.read_csv(config.outputs_tables_dir / "final_regression_results_improved.csv")
    _build_before_after_comparison(
        config=config,
        baseline_metrics=baseline_metrics,
        improved_metrics=improved_metrics,
        baseline_events=baseline_dataset,
        improved_events=improved_dataset,
        baseline_reg=baseline_reg,
        improved_reg=improved_reg,
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "run_mode",
                "sample_stocks",
                "guidance_rows_raw",
                "events_all",
                "events_baseline_after_filters",
                "events_improved_after_filters",
                "event_dataset_rows_baseline",
                "event_dataset_rows_improved",
                "period_start",
                "period_end",
            ],
            "value": [
                config.run_mode,
                len(stocks),
                len(guidance),
                len(events),
                len(baseline_events),
                len(improved_events),
                len(baseline_dataset),
                len(improved_dataset),
                config.start_date,
                config.end_date,
            ],
        }
    )
    save_csv(summary, config.outputs_tables_dir / "run_summary.csv")
    logger.info("Guidance-design pipeline complete.")
