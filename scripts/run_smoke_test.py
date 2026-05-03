import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ProjectConfig
from src.tushare_normalization import normalize_forecast, normalize_report_rc


def _ensure_non_empty(name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise RuntimeError(f"{name} returned no rows.")


def run_smoke_test() -> None:
    print("Running smoke test...")

    ProjectConfig()
    print("OK: config creation successful")

    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"

    report_rc_path = fixtures_dir / "sample_report_rc.csv"
    if report_rc_path.exists():
        df_rc = pd.read_csv(report_rc_path)
        norm_rc = normalize_report_rc(df_rc)
        print(f"OK: normalize_report_rc successful. Rows: {len(norm_rc)}")
        _ensure_non_empty("normalize_report_rc", norm_rc)

    forecast_path = fixtures_dir / "sample_forecast.csv"
    if forecast_path.exists():
        df_fc = pd.read_csv(forecast_path)
        norm_fc = normalize_forecast(df_fc)
        print(f"OK: normalize_forecast successful. Rows: {len(norm_fc)}")
        _ensure_non_empty("normalize_forecast", norm_fc)

    print("Smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
