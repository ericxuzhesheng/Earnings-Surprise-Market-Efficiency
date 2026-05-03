# Local Validation Guide

This document explains how to run the full Tushare-backed empirical pipeline on your local machine.

## Prerequisites
- A valid **Tushare Pro** token (register at [tushare.pro](https://tushare.pro/)).
- Python 3.9+ environment with dependencies installed (`pip install -r requirements.txt`).

## Step-by-Step Instructions

### 1. Set up your environment variables
You need to provide your Tushare token to the application. There are two ways to do this:

#### Method A: Using a `.env` file (Recommended)
1. Copy the template file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and replace `your_tushare_token_here` with your real token.
3. The `.env` file is ignored by git to keep your token safe.

#### Method B: Temporary environment variable
**Windows PowerShell**:
```powershell
$env:TUSHARE_TOKEN="your_token_here"
```

**Linux / macOS / WSL Bash**:
```bash
export TUSHARE_TOKEN="your_token_here"
```

### 2. Run the Full Validation
Execute the validation script which will verify your token and run the `main.py` pipeline.
```bash
python scripts/run_full_validation.py
```

### 3. Update README Results
Once the pipeline completes and generates CSVs in `outputs/tables/`, you can automatically update the "Key Results Snapshot" in the README:
```bash
python scripts/update_readme_results.py
```

### 4. Verify Success
Run the tests to ensure everything is still functioning correctly:
```bash
pytest
```

## Safety Precautions
- **Never commit your `.env` file.** It is already in `.gitignore`, but double-check before pushing.
- **Do not print your token** in any scripts or logs that might be committed.
- If you encounter API rate limits, adjust `REQUEST_PAUSE_SEC` in `.env` or `src/config.py`.

## Troubleshooting
- **Empty CSVs**: If the generated CSVs are small (e.g., 5 bytes), it usually means Tushare returned no data or your token lacks permissions for specific endpoints (like `forecast_vip`).
- **Missing Columns**: Ensure you are using a recent version of Tushare Pro.
