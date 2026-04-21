from __future__ import annotations

import pandas as pd

from src.config import ProjectConfig
from src.data_collection import DataCollector
from src.guidance_design import save_core_outputs
from src.io_utils import ensure_directories, save_csv, save_text
from src.logger_utils import setup_logger
from src.panel_outputs import save_tushare_outputs
from src.tushare_event_design import (
    apply_event_filters,
    build_legacy_guidance_panel,
    build_tushare_event_panel,
    build_tushare_events,
)
from src.tushare_normalization import (
    normalize_daily_basic,
    normalize_express,
    normalize_fina_indicator,
    normalize_forecast,
    normalize_report_rc,
)


def _save_normalized_outputs(config: ProjectConfig, normalized: dict[str, pd.DataFrame]) -> None:
    save_csv(normalized["report_rc"], config.data_processed_normalized_dir / "report_rc_normalized.csv")
    save_csv(normalized["forecast"], config.data_processed_normalized_dir / "forecast_normalized.csv")
    save_csv(normalized["express"], config.data_processed_normalized_dir / "express_normalized.csv")
    save_csv(normalized["fina_indicator"], config.data_processed_normalized_dir / "fina_indicator_normalized.csv")
    save_csv(normalized["daily_basic"], config.data_processed_normalized_dir / "daily_basic_normalized.csv")


def run_pipeline() -> None:
    config = ProjectConfig()
    ensure_directories(
        [
            config.data_raw_dir,
            config.data_raw_tushare_dir,
            config.data_processed_dir,
            config.data_processed_normalized_dir,
            config.data_processed_expectations_dir,
            config.data_processed_events_dir,
            config.data_processed_panels_dir,
            config.outputs_dir,
            config.outputs_figures_dir,
            config.outputs_tables_dir,
            config.outputs_audit_dir,
            config.logs_dir,
        ]
    )
    logger = setup_logger(config.logs_dir)
    logger.info(
        "Pipeline start | framework=%s | RUN_MODE=%s | period=%s-%s",
        config.framework_mode,
        config.run_mode,
        config.start_date,
        config.end_date,
    )

    collector = DataCollector(config=config, logger=logger)
    bundle = collector.collect_all()

    normalized = {
        "report_rc": normalize_report_rc(bundle.report_rc),
        "forecast": normalize_forecast(bundle.forecast),
        "express": normalize_express(bundle.express),
        "fina_indicator": normalize_fina_indicator(bundle.fina_indicator),
        "daily_basic": normalize_daily_basic(bundle.daily_basic),
    }
    _save_normalized_outputs(config, normalized)

    summary_rows: list[dict[str, object]] = [
        {"metric": "run_mode", "value": config.run_mode},
        {"metric": "framework_mode", "value": config.framework_mode},
        {"metric": "sample_stocks", "value": len(bundle.stocks)},
        {"metric": "forecast_rows_raw", "value": len(bundle.forecast)},
        {"metric": "express_rows_raw", "value": len(bundle.express)},
        {"metric": "fina_indicator_rows_raw", "value": len(bundle.fina_indicator)},
        {"metric": "report_rc_rows_raw", "value": len(bundle.report_rc)},
        {"metric": "period_start", "value": config.start_date},
        {"metric": "period_end", "value": config.end_date},
    ]

    if config.run_tushare_first:
        tushare_events, expectation_audit = build_tushare_events(
            stocks_df=bundle.stocks,
            market_df=bundle.market,
            forecast_df=normalized["forecast"],
            express_df=normalized["express"],
            fina_df=normalized["fina_indicator"],
            report_rc_df=normalized["report_rc"],
            config=config,
        )
        save_csv(tushare_events, config.data_processed_events_dir / "event_master_tushare_first.csv")
        save_csv(expectation_audit, config.data_processed_expectations_dir / "expectation_match_audit_tushare_first.csv")

        filtered_tushare_events = apply_event_filters(
            events_df=tushare_events,
            prices_df=bundle.prices,
            daily_basic_df=normalized["daily_basic"],
            market_df=bundle.market,
            config=config,
        )
        save_csv(filtered_tushare_events, config.data_processed_events_dir / "event_master_tushare_first_filtered.csv")

        tushare_panel, tushare_paths = build_tushare_event_panel(
            events_df=filtered_tushare_events,
            prices_df=bundle.prices,
            market_df=bundle.market,
            daily_basic_df=normalized["daily_basic"],
            config=config,
        )
        save_csv(tushare_panel, config.data_processed_panels_dir / "event_panel_tushare_first.csv")
        save_csv(tushare_paths, config.data_processed_panels_dir / "event_paths_tushare_first.csv")

        tushare_metrics = save_tushare_outputs(
            event_df=tushare_panel,
            path_df=tushare_paths,
            outputs_tables_dir=config.outputs_tables_dir,
            outputs_audit_dir=config.outputs_audit_dir,
            scenario_name="tushare_first",
        )
        summary_rows.extend(
            [
                {"metric": "tushare_events_all", "value": len(tushare_events)},
                {"metric": "tushare_events_filtered", "value": len(filtered_tushare_events)},
                {"metric": "tushare_panel_rows", "value": len(tushare_panel)},
                {"metric": "tushare_headline_coef", "value": tushare_metrics.get("headline_coef")},
                {"metric": "tushare_headline_p_value", "value": tushare_metrics.get("headline_p_value")},
            ]
        )

    if config.run_legacy_guidance:
        legacy_events, legacy_dataset, legacy_paths = build_legacy_guidance_panel(
            guidance_df=normalized["forecast"],
            stocks_df=bundle.stocks,
            prices_df=bundle.prices,
            market_df=bundle.market,
            daily_basic_df=normalized["daily_basic"],
            logger=logger,
        )
        save_csv(legacy_events, config.data_processed_events_dir / "guidance_events_legacy.csv")
        save_csv(legacy_dataset, config.data_processed_panels_dir / "event_dataset_legacy_guidance.csv")
        save_csv(legacy_paths, config.data_processed_panels_dir / "event_paths_legacy_guidance.csv")
        legacy_metrics = save_core_outputs(
            event_df=legacy_dataset,
            path_df=legacy_paths,
            outputs_tables_dir=config.outputs_tables_dir,
            outputs_figures_dir=config.outputs_figures_dir,
            logger=logger,
            scenario_name="legacy_guidance",
            primary_car="CAR60",
            car_windows=(20, 60),
            signal_col="earnings_surprise",
            use_panel_regression=False,
        )
        summary_rows.extend(
            [
                {"metric": "legacy_guidance_events", "value": len(legacy_events)},
                {"metric": "legacy_guidance_panel_rows", "value": len(legacy_dataset)},
                {"metric": "legacy_headline_coef", "value": legacy_metrics.get("coef")},
                {"metric": "legacy_headline_p_value", "value": legacy_metrics.get("p_value")},
            ]
        )

    summary = pd.DataFrame(summary_rows)
    save_csv(summary, config.outputs_tables_dir / "run_summary.csv")

    note_lines = [
        "Tushare-first pipeline update note",
        f"Framework mode: {config.framework_mode}",
        "The preferred path now uses report_rc for sell-side expectations, forecast/express/fina_indicator for event types, and daily_basic for controls.",
        "Legacy guidance-only outputs remain runnable as fallback and comparison.",
    ]
    save_text("\n".join(note_lines) + "\n", config.outputs_audit_dir / "tushare_first_update_note.txt")
    logger.info("Pipeline complete.")
