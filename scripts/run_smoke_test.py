import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tushare_normalization import normalize_report_rc, normalize_forecast
from src.config import ProjectConfig

def run_smoke_test():
    print("Running smoke test...")
    
    # 1. Test Config creation
    config = ProjectConfig()
    print("✓ Config creation successful")
    
    # 2. Test Normalization with fixtures
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    
    report_rc_path = fixtures_dir / "sample_report_rc.csv"
    if report_rc_path.exists():
        df_rc = pd.read_csv(report_rc_path)
        norm_rc = normalize_report_rc(df_rc)
        print(f"✓ normalize_report_rc successful. Rows: {len(norm_rc)}")
        assert not norm_rc.empty
    
    forecast_path = fixtures_dir / "sample_forecast.csv"
    if forecast_path.exists():
        df_fc = pd.read_csv(forecast_path)
        norm_fc = normalize_forecast(df_fc)
        print(f"✓ normalize_forecast successful. Rows: {len(norm_fc)}")
        assert not norm_fc.empty

    print("\nSmoke test passed!")

if __name__ == "__main__":
    run_smoke_test()
