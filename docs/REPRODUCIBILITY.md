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
   Register at [Tushare Pro](https://tushare.pro/) and set your token:
   ```bash
   export TUSHARE_TOKEN="your_token_here" # Windows: set TUSHARE_TOKEN=your_token_here
   ```

## Running the Pipeline
- **Full Run**: `python main.py`
- **Smoke Test**: `python scripts/run_smoke_test.py` (No token required, uses fixtures).
- **Unit Tests**: `pytest` (If installed).

## Expected Outputs
- CSV tables in `outputs/tables/`.
- Figures in `outputs/figures/`.
- Audit logs in `outputs/audit/`.

## Known Non-Reproducibility Sources
- **Tushare Permissions**: Some endpoints (like `forecast_vip`) require specific credit levels (积分). If your account lacks these, the sample size will be smaller.
- **Update Lag**: Tushare data is updated daily. Running the pipeline on different dates may result in slight variations in recent observations.
- **Analyst Coverage**: The `report_rc` endpoint has finite history; results may differ if Tushare truncates older data.
