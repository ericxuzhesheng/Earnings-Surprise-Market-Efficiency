import os
import sys
import subprocess
from pathlib import Path

def run_full_validation():
    print("Checking TUSHARE_TOKEN...")
    token = os.getenv("TUSHARE_TOKEN")
    
    if not token or token == "your_tushare_token_here":
        print("ERROR: TUSHARE_TOKEN is missing or not set correctly.")
        print("Please set it via 'export TUSHARE_TOKEN=...' or in a .env file.")
        sys.exit(1)
    
    print("TUSHARE_TOKEN found (length: {}).".format(len(token)))
    
    # Check if main.py exists
    main_path = Path(__file__).parent.parent / "main.py"
    if not main_path.exists():
        print("ERROR: main.py not found at {}".format(main_path))
        sys.exit(1)
        
    print("Running the real pipeline (main.py)...")
    try:
        # We run it as a subprocess to keep environment isolation if needed
        # and to capture output cleanly.
        result = subprocess.run([sys.executable, str(main_path)], check=True)
        if result.returncode == 0:
            print("\n✓ Pipeline completed successfully.")
        else:
            print("\n× Pipeline failed with return code {}.".format(result.returncode))
            sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print("\n× Pipeline failed with error: {}".format(e))
        sys.exit(1)

    # Verify expected output CSVs
    print("\nVerifying expected outputs...")
    output_dir = Path(__file__).parent.parent / "outputs" / "tables"
    expected_files = [
        "sample_construction.csv",
        "headline_signal_comparison.csv",
        "event_window_car_summary.csv",
        "matching_quality_diagnostics.csv"
    ]
    
    missing = []
    for f in expected_files:
        if not (output_dir / f).exists():
            missing.append(f)
        else:
            print("✓ Found {}".format(f))
            
    if missing:
        print("\nWARNING: Some expected output files are missing:")
        for m in missing:
            print("  - {}".format(m))
    else:
        print("\nValidation Summary: All core output tables generated.")

if __name__ == "__main__":
    run_full_validation()
