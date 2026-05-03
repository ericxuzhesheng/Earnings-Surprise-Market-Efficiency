import os
import subprocess
import sys
from pathlib import Path


def run_full_validation() -> None:
    print("Checking TUSHARE_TOKEN...")
    token = os.getenv("TUSHARE_TOKEN")

    if not token or token == "your_tushare_token_here":
        print("ERROR: TUSHARE_TOKEN is missing or not set correctly.")
        print("Please set it via 'export TUSHARE_TOKEN=...' or in a .env file.")
        sys.exit(1)

    print(f"OK: TUSHARE_TOKEN found (length: {len(token)}).")

    main_path = Path(__file__).parent.parent / "main.py"
    if not main_path.exists():
        print(f"ERROR: main.py not found at {main_path}")
        sys.exit(1)

    print("Running the real pipeline (main.py)...")
    try:
        subprocess.run([sys.executable, str(main_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        sys.exit(e.returncode)

    print("\nOK: Pipeline completed successfully.")
    print("\nVerifying expected outputs...")
    output_dir = Path(__file__).parent.parent / "outputs" / "tables"
    expected_files = [
        "sample_construction.csv",
        "headline_signal_comparison.csv",
        "event_window_car_summary.csv",
        "matching_quality_diagnostics.csv",
    ]

    missing = []
    for f in expected_files:
        if not (output_dir / f).exists():
            missing.append(f)
        else:
            print(f"OK: Found {f}")

    if missing:
        print("\nERROR: Some expected output files are missing:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("\nValidation summary: All core output tables generated.")


if __name__ == "__main__":
    run_full_validation()
