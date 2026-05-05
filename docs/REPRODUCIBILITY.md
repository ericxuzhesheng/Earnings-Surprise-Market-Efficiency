# Reproducibility Guide

## Environment Setup
1. **Python Version**: 3.9+ recommended.
2. **Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Tushare Token**:
   Register at [Tushare Pro](https://tushare.pro/) and set your token. You can either export it in your shell or place it in the repository-root `.env` file:
   ```bash
   export TUSHARE_TOKEN="your_token_here" # Windows: set TUSHARE_TOKEN=your_token_here
   ```

## Validation Status
- **Last Validation Run**: 2026-05-03
- **Environment**: win32
- **Tushare Token**: Missing (Validation run used fixture data only).
- **Status**: 
    - `py_compile`: PASSED
    - `scripts/run_smoke_test.py`: PASSED (Fixture data)
    - `pytest`: OUTDATED, rerun after code changes
    - `main.py`: NOT RUN (Requires TUSHARE_TOKEN)

## Expected Outputs
- CSV tables in `outputs/tables/`.
- Figures in `outputs/figures/`.
- Audit logs in `outputs/audit/`.

## Cache Behavior

- **Per-stock caching**: Raw API responses are cached under `data_raw/cache/` organized by endpoint and cache key.
- **Date-range keys**: Cache keys for date-dependent endpoints (prices, daily_basic, fina_indicator, forecast, express) now include `start_date` and `end_date`. Changing the date range in `.env` will automatically fetch data for the new period while reusing cached data for the old period.
- **Force refresh**: Set `FORCE_REFRESH=1` in `.env` or as an environment variable to bypass all caches and re-fetch everything from Tushare/Akshare APIs.
- **Cache location**: `data_raw/cache/{endpoint}/{cache_key}.csv`

## Known Non-Reproducibility Sources
- **Tushare Permissions**: Some endpoints (like `forecast_vip`) require specific credit levels (积分). If your account lacks these, the sample size will be smaller.
- **Rate Limiting**: `fina_indicator` is limited to ~500 calls/day. Full 1200-stock runs will hit this limit; subsequent stocks fall back to cached data from prior runs.
- **Update Lag**: Tushare data is updated daily. Running the pipeline on different dates may result in slight variations in recent observations.
- **Analyst Coverage**: The `report_rc` endpoint has finite history; results may differ if Tushare truncates older data.
- **Cache Staleness**: Orphaned cache files from prior date ranges are harmless but may accumulate. Delete `data_raw/cache/` subdirectories to clean up.
