# Data Dictionary

## Tushare Tables Used
- `stock_basic`: Base information for stocks (industry, list date).
- `daily_basic`: Market valuation and liquidity (MV, PE, Turnover).
- `forecast` / `forecast_vip`: Management earnings guidance.
- `express` / `express_vip`: Preliminary earnings results.
- `fina_indicator`: Financial ratios and key metrics.
- `report_rc`: Sell-side analyst research report forecasts.
- `daily`: Stock price and return data.
- `index_daily`: Market index returns (CSI 300).

## Key Fields in Processed Datasets
- `main_surprise_raw`: $Actual - Expected$
- `main_surprise_pct`: Proportional surprise.
- `main_surprise_std`: Cross-sectionally standardized surprise.
- `report_rc_match_tier`: The quality level of the expectation match.
- `CAR_x_y`: Cumulative Abnormal Return from day $x$ to day $y$.

## Generated Intermediate Files
- `data_processed/events/event_master_tushare_first.csv`: Raw matched events.
- `data_processed/panels/event_panel_tushare_first.csv`: Final filtered regression panel.

## Output Files (`outputs/tables/`)
- `sample_construction.csv`: Step-by-step filter counts.
- `headline_signal_comparison.csv`: Regression results across surprise metrics.
- `event_window_car_summary.csv`: Average CARs across all windows.
- `leakage_drift_diagnostics.csv`: Comparison of pre- and post-event returns.
- `matching_quality_diagnostics.csv`: Statistics on expectation matching.
