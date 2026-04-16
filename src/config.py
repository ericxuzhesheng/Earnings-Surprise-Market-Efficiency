from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class ProjectConfig:
    run_mode: str = os.getenv("RUN_MODE", "test").lower()  # "test" or "full"
    tushare_token: str = os.getenv(
        "TUSHARE_TOKEN",
        "0d56d01072fc65f04db23d9ebc89ddba410e6e2614f8280962e35c32",
    )
    market_index_code_tushare: str = "399300.SZ"  # CSI 300
    market_index_symbol_akshare: str = "sh000300"
    start_date_test: str = "20200101"
    end_date_test: str = "20260416"
    start_date_full: str = "20200101"
    end_date_full: str = "20260416"
    sample_stock_count_test: int = int(os.getenv("SAMPLE_STOCK_COUNT_TEST", "0"))
    sample_stock_count_full: int = int(os.getenv("SAMPLE_STOCK_COUNT_FULL", "1200"))
    min_events_per_stock: int = 2
    beta_estimation_window: int = 120
    beta_buffer_days: int = 20
    event_windows: tuple = (20, 40, 60)
    quantile_groups: int = 5
    min_obs_regression: int = 30
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    use_spec_search: bool = True
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_wait_seconds: float = float(os.getenv("RETRY_WAIT_SECONDS", "0.8"))
    use_cache: bool = os.getenv("USE_CACHE", "1") == "1"
    force_refresh: bool = os.getenv("FORCE_REFRESH", "0") == "1"
    enable_cross_check: bool = os.getenv("ENABLE_CROSS_CHECK", "0") == "1"
    cross_check_sample_count: int = int(os.getenv("CROSS_CHECK_SAMPLE_COUNT", "3"))
    max_workers_test: int = int(os.getenv("MAX_WORKERS_TEST", "8"))
    max_workers_full: int = int(os.getenv("MAX_WORKERS_FULL", "16"))
    request_retry: int = int(os.getenv("REQUEST_RETRY", "3"))
    request_pause_sec: float = float(os.getenv("REQUEST_PAUSE_SEC", "0.1"))
    liquidity_turnover20_old: float = 0.5
    liquidity_turnover20_new: float = float(os.getenv("LIQUIDITY_TURNOVER20_NEW", "0.3"))

    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_raw_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    outputs_figures_dir: Path = field(init=False)
    outputs_tables_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_raw_dir = self.project_root / "data_raw"
        self.data_processed_dir = self.project_root / "data_processed"
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_figures_dir = self.outputs_dir / "figures"
        self.outputs_tables_dir = self.outputs_dir / "tables"
        self.logs_dir = self.project_root / "logs"

    @property
    def start_date(self) -> str:
        return self.start_date_test if self.run_mode == "test" else self.start_date_full

    @property
    def end_date(self) -> str:
        return self.end_date_test if self.run_mode == "test" else self.end_date_full

    @property
    def sample_stock_count(self) -> int | None:
        val = (
            self.sample_stock_count_test
            if self.run_mode == "test"
            else self.sample_stock_count_full
        )
        return None if val <= 0 else val

    @property
    def max_workers(self) -> int:
        return self.max_workers_test if self.run_mode == "test" else self.max_workers_full
