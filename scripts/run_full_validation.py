import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env_utils import load_local_env


def _validate_csv_output(path: Path, required_columns: list[str] | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing output file: {path.name}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Output file is empty: {path.name}")

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Output file {path.name} is missing columns: {missing}")


def run_full_validation() -> None:
    load_local_env(Path(__file__).parent.parent)
    print("Checking TUSHARE_TOKEN...")
    token = os.getenv("TUSHARE_TOKEN")

    if not token or token == "your_tushare_token_here":
        print("ERROR: TUSHARE_TOKEN is missing or not set correctly.")
        print("Please set it in your shell or in the repository .env file.")
        sys.exit(1)

    print(f"OK: TUSHARE_TOKEN found (length: {len(token)}).")

    env = os.environ.copy()
    env["RUN_MODE"] = "full"
    env.setdefault("FRAMEWORK_MODE", "tushare_first")

    print("Effective settings for this run:")
    print(f"  RUN_MODE       = {env['RUN_MODE']}")
    print(f"  FRAMEWORK_MODE = {env['FRAMEWORK_MODE']}")
    print(f"  START_DATE_FULL = {env.get('START_DATE_FULL', '(default)')}")
    print(f"  END_DATE_FULL   = {env.get('END_DATE_FULL', '(default)')}")
    print(f"  SAMPLE_STOCK_COUNT_FULL = {env.get('SAMPLE_STOCK_COUNT_FULL', '(default)')}")
    print(f"  USE_CACHE      = {env.get('USE_CACHE', '(default)')}")
    print(f"  FORCE_REFRESH  = {env.get('FORCE_REFRESH', '(default)')}")

    main_path = Path(__file__).parent.parent / "main.py"
    if not main_path.exists():
        print(f"ERROR: main.py not found at {main_path}")
        sys.exit(1)

    print("Running the real pipeline (main.py)...")
    try:
        subprocess.run([sys.executable, str(main_path)], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        sys.exit(e.returncode)

    print("\nOK: Pipeline completed successfully.")
    print("\nVerifying expected outputs...")
    output_dir = Path(__file__).parent.parent / "outputs" / "tables"
    expected_files = {
        "sample_construction.csv": ["stage", "event_count"],
        "headline_signal_comparison.csv": ["signal_scale"],
        "event_window_car_summary.csv": ["mean_car"],
        "matching_quality_diagnostics.csv": ["benchmark_method"],
        "run_summary.csv": ["metric", "value"],
        "event_dataset_tushare_first.csv": None,
        "headline_signal_comparison_tushare_first.csv": None,
        "ablation_catalog_tushare_first.csv": None,
        "ablation_results_tushare_first.csv": None,
    }

    try:
        for filename, required_columns in expected_files.items():
            _validate_csv_output(output_dir / filename, required_columns)
            print(f"OK: Validated {filename}")
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        print(f"\nERROR: Output validation failed: {exc}")
        sys.exit(1)

    print("\nValidation summary: All core output tables generated.")


if __name__ == "__main__":
    run_full_validation()
