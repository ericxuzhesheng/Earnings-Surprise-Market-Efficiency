from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class ProjectConfig:
    run_mode: str = os.getenv("RUN_MODE", "test").lower()
    framework_mode: str = os.getenv("FRAMEWORK_MODE", "tushare_first").lower()
    tushare_token: str = os.getenv("TUSHARE_TOKEN", "")

    market_index_code_tushare: str = os.getenv("MARKET_INDEX_CODE_TUSHARE", "399300.SZ")
    market_index_symbol_akshare: str = os.getenv("MARKET_INDEX_SYMBOL_AKSHARE", "sh000300")

    start_date_test: str = os.getenv("START_DATE_TEST", "20200101")
    end_date_test: str = os.getenv("END_DATE_TEST", "20260421")
    start_date_full: str = os.getenv("START_DATE_FULL", "20200101")
    end_date_full: str = os.getenv("END_DATE_FULL", "20260421")

    sample_stock_count_test: int = int(os.getenv("SAMPLE_STOCK_COUNT_TEST", "300"))
    sample_stock_count_full: int = int(os.getenv("SAMPLE_STOCK_COUNT_FULL", "1200"))
    min_events_per_stock: int = int(os.getenv("MIN_EVENTS_PER_STOCK", "2"))

    beta_estimation_window: int = int(os.getenv("BETA_ESTIMATION_WINDOW", "120"))
    beta_buffer_days: int = int(os.getenv("BETA_BUFFER_DAYS", "20"))
    event_windows: tuple[int, ...] = field(
        default_factory=lambda: tuple(
            int(x.strip())
            for x in os.getenv("EVENT_WINDOWS", "3,5,10,20,60").split(",")
            if x.strip()
        )
    )
    quantile_groups: int = int(os.getenv("QUANTILE_GROUPS", "5"))
    min_obs_regression: int = int(os.getenv("MIN_OBS_REGRESSION", "30"))
    winsor_lower: float = float(os.getenv("WINSOR_LOWER", "0.01"))
    winsor_upper: float = float(os.getenv("WINSOR_UPPER", "0.99"))

    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_wait_seconds: float = float(os.getenv("RETRY_WAIT_SECONDS", "0.8"))
    use_cache: bool = os.getenv("USE_CACHE", "1") == "1"
    force_refresh: bool = os.getenv("FORCE_REFRESH", "0") == "1"
    max_workers_test: int = int(os.getenv("MAX_WORKERS_TEST", "8"))
    max_workers_full: int = int(os.getenv("MAX_WORKERS_FULL", "16"))
    request_retry: int = int(os.getenv("REQUEST_RETRY", "3"))
    request_pause_sec: float = float(os.getenv("REQUEST_PAUSE_SEC", "0.12"))

    use_report_rc: bool = os.getenv("USE_REPORT_RC", "1") == "1"
    use_forecast_vip: bool = os.getenv("USE_FORECAST_VIP", "1") == "1"
    use_express_vip: bool = os.getenv("USE_EXPRESS_VIP", "1") == "1"
    use_fina_indicator: bool = os.getenv("USE_FINA_INDICATOR", "1") == "1"

    report_freshness_days: int = int(os.getenv("REPORT_FRESHNESS_DAYS", "365"))
    min_valid_report_count: int = int(os.getenv("MIN_VALID_REPORT_COUNT", "1"))
    consensus_method: str = os.getenv("CONSENSUS_METHOD", "median").lower()
    consensus_value_field: str = os.getenv("CONSENSUS_VALUE_FIELD", "np_first").lower()
    fallback_expectation_mode: str = os.getenv("FALLBACK_EXPECTATION_MODE", "legacy_proxy").lower()

    min_listed_trading_days: int = int(os.getenv("MIN_LISTED_TRADING_DAYS", "120"))
    turnover20_threshold: float = float(os.getenv("TURNOVER20_THRESHOLD", "0.3"))
    liquidity_turnover20_old: float = float(os.getenv("LIQUIDITY_TURNOVER20_OLD", "0.5"))
    liquidity_turnover20_new: float = float(os.getenv("LIQUIDITY_TURNOVER20_NEW", "0.3"))
    min_positive_volume: float = float(os.getenv("MIN_POSITIVE_VOLUME", "0"))
    save_audit_outputs: bool = os.getenv("SAVE_AUDIT_OUTPUTS", "1") == "1"
    run_legacy_fallback: bool = os.getenv("RUN_LEGACY_FALLBACK", "1") == "1"

    project_root: Path = field(default_factory=lambda: Path.cwd())
    data_raw_dir: Path = field(init=False)
    data_raw_tushare_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)
    data_processed_normalized_dir: Path = field(init=False)
    data_processed_expectations_dir: Path = field(init=False)
    data_processed_events_dir: Path = field(init=False)
    data_processed_panels_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    outputs_figures_dir: Path = field(init=False)
    outputs_tables_dir: Path = field(init=False)
    outputs_audit_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_raw_dir = self.project_root / "data_raw"
        self.data_raw_tushare_dir = self.data_raw_dir / "tushare"
        self.data_processed_dir = self.project_root / "data_processed"
        self.data_processed_normalized_dir = self.data_processed_dir / "normalized"
        self.data_processed_expectations_dir = self.data_processed_dir / "expectations"
        self.data_processed_events_dir = self.data_processed_dir / "events"
        self.data_processed_panels_dir = self.data_processed_dir / "panels"
        self.outputs_dir = self.project_root / "outputs"
        self.outputs_figures_dir = self.outputs_dir / "figures"
        self.outputs_tables_dir = self.outputs_dir / "tables"
        self.outputs_audit_dir = self.outputs_dir / "audit"
        self.logs_dir = self.project_root / "logs"

    @property
    def start_date(self) -> str:
        return self.start_date_test if self.run_mode == "test" else self.start_date_full

    @property
    def end_date(self) -> str:
        return self.end_date_test if self.run_mode == "test" else self.end_date_full

    @property
    def sample_stock_count(self) -> int | None:
        val = self.sample_stock_count_test if self.run_mode == "test" else self.sample_stock_count_full
        return None if val <= 0 else val

    @property
    def max_workers(self) -> int:
        return self.max_workers_test if self.run_mode == "test" else self.max_workers_full

    @property
    def run_tushare_first(self) -> bool:
        return self.framework_mode in {"tushare_first", "both"}

    @property
    def run_legacy_guidance(self) -> bool:
        return self.framework_mode in {"legacy_guidance", "both"} or self.run_legacy_fallback
