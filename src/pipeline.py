from __future__ import annotations

from dataclasses import replace

import pandas as pd

from src.config import ProjectConfig
from src.data_collection import DataCollector
from src.guidance_design import save_core_outputs
from src.io_utils import ensure_directories, save_csv, save_text
from src.logger_utils import setup_logger
from src.panel_outputs import save_tushare_outputs
from src.tushare_event_design import (
    annotate_event_filters,
    apply_event_filters,
    build_benchmark_quality_summary,
    build_event_type_signal_summary,
    build_expectation_coverage_summary,
    build_filter_funnel,
    build_legacy_guidance_panel,
    build_tushare_event_panel,
    build_tushare_events,
    build_timing_alignment_summary,
    run_ablation_specs,
    summarize_window_availability,
)
from src.tushare_normalization import (
    normalize_cninfo_preannouncement,
    normalize_daily_basic,
    normalize_eastmoney_profit_forecast,
    normalize_eastmoney_research_report,
    normalize_express,
    normalize_fina_indicator,
    normalize_forecast,
    normalize_report_rc,
)


def _build_augmentation_audit_frames(
    config: ProjectConfig,
    bundle,
    normalized: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    source_rows = [
        {
            "source_name": "cninfo_preannouncement",
            "source_tier": "event_tier_1_cninfo_official_disclosure",
            "enabled": int(config.enable_free_augmentation and config.use_cninfo_event_augmentation),
            "raw_rows": len(bundle.cninfo_preannouncement),
            "normalized_rows": len(normalized.get("cninfo_preannouncement", pd.DataFrame())),
        },
        {
            "source_name": "eastmoney_profit_forecast",
            "source_tier": "tier_2_eastmoney_profit_forecast",
            "enabled": int(config.enable_free_augmentation and config.use_eastmoney_expectation_augmentation),
            "raw_rows": len(bundle.eastmoney_profit_forecast),
            "normalized_rows": len(normalized.get("eastmoney_profit_forecast", pd.DataFrame())),
        },
        {
            "source_name": "eastmoney_research_report",
            "source_tier": "tier_3_eastmoney_research_report_text",
            "enabled": int(config.enable_free_augmentation and config.use_eastmoney_expectation_augmentation),
            "raw_rows": len(bundle.eastmoney_research_report),
            "normalized_rows": len(normalized.get("eastmoney_research_report", pd.DataFrame())),
        },
    ]
    return {
        "free_source_capabilities": bundle.free_source_capabilities.copy(),
        "augmentation_status": pd.DataFrame(
            [
                {
                    "enable_free_augmentation": int(config.enable_free_augmentation),
                    "use_cninfo_event_augmentation": int(config.use_cninfo_event_augmentation),
                    "use_eastmoney_expectation_augmentation": int(config.use_eastmoney_expectation_augmentation),
                    "strict_match_mode": config.strict_match_mode,
                    "source_tier_stack": config.source_tier_stack,
                }
            ]
        ),
        "augmentation_collection_status": pd.DataFrame(source_rows),
        "source_tier_stack": pd.DataFrame(
            [
                {"tier_order": idx + 1, "source_tier": tier.strip()}
                for idx, tier in enumerate(config.source_tier_stack.split(","))
                if tier.strip()
            ]
        ),
        "augmentation_match_modes": pd.DataFrame(
            [{"match_mode": config.strict_match_mode, "mode_group": "strict"}]
            + [{"match_mode": mode, "mode_group": "relaxed"} for mode in config.relaxed_match_modes]
        ),
    }


def _save_normalized_outputs(config: ProjectConfig, normalized: dict[str, pd.DataFrame]) -> None:
    normalized_output_paths = {
        "report_rc": config.data_processed_normalized_dir / "report_rc_normalized.csv",
        "forecast": config.data_processed_normalized_dir / "forecast_normalized.csv",
        "express": config.data_processed_normalized_dir / "express_normalized.csv",
        "fina_indicator": config.data_processed_normalized_dir / "fina_indicator_normalized.csv",
        "daily_basic": config.data_processed_normalized_dir / "daily_basic_normalized.csv",
        "cninfo_preannouncement": config.data_processed_normalized_dir / "cninfo_preannouncement_normalized.csv",
        "eastmoney_profit_forecast": config.data_processed_normalized_dir / "eastmoney_profit_forecast_normalized.csv",
        "eastmoney_research_report": config.data_processed_normalized_dir / "eastmoney_research_report_normalized.csv",
    }
    for key, path in normalized_output_paths.items():
        if key in normalized:
            save_csv(normalized[key], path)

    audit_output_paths = {
        "free_source_capabilities": config.outputs_audit_dir / "free_source_capabilities.csv",
        "augmentation_status": config.outputs_audit_dir / "augmentation_status.csv",
        "augmentation_collection_status": config.outputs_audit_dir / "augmentation_collection_status.csv",
        "source_tier_stack": config.outputs_audit_dir / "source_tier_stack_config.csv",
        "augmentation_match_modes": config.outputs_audit_dir / "augmentation_match_modes.csv",
    }
    for key, path in audit_output_paths.items():
        if key in normalized:
            save_csv(normalized[key], path)


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
        "cninfo_preannouncement": normalize_cninfo_preannouncement(bundle.cninfo_preannouncement),
        "eastmoney_profit_forecast": normalize_eastmoney_profit_forecast(bundle.eastmoney_profit_forecast),
        "eastmoney_research_report": normalize_eastmoney_research_report(bundle.eastmoney_research_report),
    }
    normalized.update(_build_augmentation_audit_frames(config, bundle, normalized))
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
        tushare_events, expectation_audit, expectation_candidates = build_tushare_events(
            stocks_df=bundle.stocks,
            market_df=bundle.market,
            forecast_df=normalized["forecast"],
            express_df=normalized["express"],
            fina_df=normalized["fina_indicator"],
            report_rc_df=normalized["report_rc"],
            config=config,
            match_tier="strict_same_quarter",
            eastmoney_profit_forecast_df=normalized["eastmoney_profit_forecast"],
            eastmoney_research_report_df=normalized["eastmoney_research_report"],
        )
        save_csv(tushare_events, config.data_processed_events_dir / "event_master_tushare_first.csv")
        save_csv(expectation_audit, config.data_processed_expectations_dir / "expectation_match_audit_tushare_first.csv")
        save_csv(expectation_candidates, config.data_processed_expectations_dir / "expectation_candidates_tushare_first.csv")

        annotated_events = annotate_event_filters(
            events_df=tushare_events,
            prices_df=bundle.prices,
            daily_basic_df=normalized["daily_basic"],
            market_df=bundle.market,
            config=config,
        )
        save_csv(annotated_events, config.data_processed_events_dir / "event_master_annotated_tushare_first.csv")

        filter_funnel_df = build_filter_funnel(annotated_events)
        benchmark_quality_df = build_benchmark_quality_summary(annotated_events)
        timing_alignment_df = build_timing_alignment_summary(annotated_events)
        expectation_coverage_df = build_expectation_coverage_summary(annotated_events)
        event_signal_df = build_event_type_signal_summary(annotated_events)
        window_availability_df = summarize_window_availability(annotated_events, bundle.prices, config)

        filtered_tushare_events = apply_event_filters(
            events_df=tushare_events,
            prices_df=bundle.prices,
            daily_basic_df=normalized["daily_basic"],
            market_df=bundle.market,
            config=config,
            profile="restrictive",
        )
        if filtered_tushare_events.empty or "event_type" not in filtered_tushare_events.columns:
            headline_tushare_events = pd.DataFrame(columns=filtered_tushare_events.columns)
        else:
            headline_tushare_events = filtered_tushare_events[filtered_tushare_events["event_type"].eq("preannouncement")].reset_index(drop=True)

        tushare_panel, tushare_paths = build_tushare_event_panel(
            events_df=headline_tushare_events,
            prices_df=bundle.prices,
            market_df=bundle.market,
            daily_basic_df=normalized["daily_basic"],
            config=config,
        )
        supplementary_panel, supplementary_paths = build_tushare_event_panel(
            events_df=filtered_tushare_events,
            prices_df=bundle.prices,
            market_df=bundle.market,
            daily_basic_df=normalized["daily_basic"],
            config=config,
        )
        save_csv(tushare_panel, config.data_processed_panels_dir / "event_panel_tushare_first.csv")
        save_csv(tushare_paths, config.data_processed_panels_dir / "event_paths_tushare_first.csv")
        save_csv(supplementary_panel, config.data_processed_panels_dir / "event_panel_tushare_first_all_events.csv")
        save_csv(supplementary_paths, config.data_processed_panels_dir / "event_paths_tushare_first_all_events.csv")

        diagnostic_panels: list[pd.DataFrame] = []
        for consensus_method in ["latest_snapshot", "latest_per_analyst", "pooled_median"]:
            diagnostic_config = replace(config, consensus_method=consensus_method)
            for match_tier in [
                "strict_same_quarter",
                "same_fiscal_year_nearest_valid",
                "latest_valid_pre_event",
                "multi_report_median",
            ]:
                diag_events, _, _ = build_tushare_events(
                    stocks_df=bundle.stocks,
                    market_df=bundle.market,
                    forecast_df=normalized["forecast"],
                    express_df=normalized["express"],
                    fina_df=normalized["fina_indicator"],
                    report_rc_df=normalized["report_rc"],
                    config=diagnostic_config,
                    match_tier=match_tier,
                    eastmoney_profit_forecast_df=normalized["eastmoney_profit_forecast"],
                    eastmoney_research_report_df=normalized["eastmoney_research_report"],
                )
                diag_events = annotate_event_filters(
                    events_df=diag_events,
                    prices_df=bundle.prices,
                    daily_basic_df=normalized["daily_basic"],
                    market_df=bundle.market,
                    config=diagnostic_config,
                )
                diagnostic_panel_single, _ = build_tushare_event_panel(
                    events_df=diag_events,
                    prices_df=bundle.prices,
                    market_df=bundle.market,
                    daily_basic_df=normalized["daily_basic"],
                    config=diagnostic_config,
                )
                if not diagnostic_panel_single.empty:
                    diagnostic_panel_single["diagnostic_consensus_method"] = consensus_method
                    diagnostic_panel_single["diagnostic_match_tier"] = match_tier
                    diagnostic_panels.append(diagnostic_panel_single)

        diagnostic_panel = pd.concat(diagnostic_panels, ignore_index=True) if diagnostic_panels else pd.DataFrame()
        save_csv(diagnostic_panel, config.data_processed_panels_dir / "event_panel_diagnostic_tushare_first.csv")

        ablation_catalog_df, ablation_results_df = run_ablation_specs(diagnostic_panel, config)

        tushare_metrics = save_tushare_outputs(
            event_df=tushare_panel,
            path_df=tushare_paths,
            outputs_tables_dir=config.outputs_tables_dir,
            outputs_audit_dir=config.outputs_audit_dir,
            scenario_name="tushare_first",
            diagnostic_panel_df=diagnostic_panel,
            supplementary_event_df=supplementary_panel,
            supplementary_path_df=supplementary_paths,
            filter_funnel_df=filter_funnel_df,
            benchmark_quality_df=benchmark_quality_df,
            timing_alignment_df=timing_alignment_df,
            expectation_coverage_df=expectation_coverage_df,
            event_signal_df=event_signal_df,
            ablation_catalog_df=ablation_catalog_df,
            ablation_results_df=ablation_results_df,
            window_availability_df=window_availability_df,
        )
        save_csv(
            pd.DataFrame([
                {"sample_name": "headline_preannouncement_only", "event_count": len(headline_tushare_events)},
                {"sample_name": "supplementary_all_event_types", "event_count": len(filtered_tushare_events)},
            ]),
            config.outputs_tables_dir / "headline_sample_comparison_tushare_first.csv",
        )
        save_csv(
            pd.DataFrame([
                {"sample_name": "headline_preannouncement_only", "panel_rows": len(tushare_panel)},
                {"sample_name": "supplementary_all_event_types", "panel_rows": len(supplementary_panel)},
            ]),
            config.outputs_audit_dir / "headline_panel_coverage_tushare_first.csv",
        )

        summary_rows.extend(
            [
                {"metric": "tushare_events_all", "value": len(tushare_events)},
                {"metric": "tushare_events_filtered", "value": len(filtered_tushare_events)},
                {"metric": "tushare_headline_events", "value": len(headline_tushare_events)},
                {"metric": "tushare_panel_rows", "value": len(tushare_panel)},
                {"metric": "tushare_headline_sample", "value": "preannouncement_only"},
                {"metric": "tushare_headline_coef", "value": tushare_metrics.get("headline_coef")},
                {"metric": "tushare_headline_p_value", "value": tushare_metrics.get("headline_p_value")},
                {"metric": "usable_signal_rows", "value": tushare_metrics.get("usable_signal_rows")},
                {"metric": "strict_match_rows", "value": tushare_metrics.get("strict_match_rows")},
                {"metric": "headline_signal_scale", "value": tushare_metrics.get("headline_signal_scale")},
                {"metric": "headline_match_tier", "value": tushare_metrics.get("headline_match_tier")},
                {"metric": "diagnostic_recommendation", "value": tushare_metrics.get("diagnostic_recommendation")},
                {"metric": "strongest_spec_id", "value": tushare_metrics.get("strongest_spec_id")},
                {"metric": "strongest_spec_window", "value": tushare_metrics.get("strongest_spec_window")},
                {"metric": "strongest_spec_coef", "value": tushare_metrics.get("strongest_spec_coef")},
                {"metric": "strongest_spec_p_value", "value": tushare_metrics.get("strongest_spec_p_value")},
                {"metric": "strongest_spec_nobs", "value": tushare_metrics.get("strongest_spec_nobs")},
            ]
        )

        save_csv(pd.DataFrame(summary_rows), config.outputs_tables_dir / "run_summary.csv")
        save_csv(
            pd.DataFrame(
                [
                    {
                        "note": "Tushare-first repo now treats preannouncement_only as the headline sample and all-event results as supplementary diagnostics.",
                        "headline_signal_scale": tushare_metrics.get("headline_signal_scale"),
                        "headline_match_tier": tushare_metrics.get("headline_match_tier"),
                        "diagnostic_recommendation": tushare_metrics.get("diagnostic_recommendation"),
                    }
                ]
            ),
            config.outputs_audit_dir / "tushare_first_update_note.csv",
        )

        summary_rows.extend(
            [
                {"metric": "tushare_events_all", "value": len(tushare_events)},
                {"metric": "tushare_events_filtered", "value": len(filtered_tushare_events)},
                {"metric": "tushare_panel_rows", "value": len(tushare_panel)},
                {"metric": "tushare_headline_coef", "value": tushare_metrics.get("headline_coef")},
                {"metric": "tushare_headline_p_value", "value": tushare_metrics.get("headline_p_value")},
                {"metric": "usable_signal_rows", "value": tushare_metrics.get("usable_signal_rows")},
                {"metric": "strict_match_rows", "value": tushare_metrics.get("strict_match_rows")},
                {"metric": "diagnostic_recommendation", "value": tushare_metrics.get("diagnostic_recommendation")},
                {"metric": "strongest_spec_id", "value": tushare_metrics.get("strongest_spec_id")},
                {"metric": "strongest_spec_window", "value": tushare_metrics.get("strongest_spec_window")},
                {"metric": "strongest_spec_coef", "value": tushare_metrics.get("strongest_spec_coef")},
                {"metric": "strongest_spec_p_value", "value": tushare_metrics.get("strongest_spec_p_value")},
                {"metric": "strongest_spec_nobs", "value": tushare_metrics.get("strongest_spec_nobs")},
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
