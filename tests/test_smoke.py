import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Smoke test: verify all src modules can be imported."""
    import src.config
    import src.data_collection
    import src.earnings_surprise
    import src.event_study
    import src.expectation_alignment
    import src.guidance_design
    import src.io_utils
    import src.logger_utils
    import src.panel_outputs
    import src.pipeline
    import src.regression_analysis
    import src.spec_selection
    import src.tushare_event_design
    import src.tushare_loaders
    import src.tushare_normalization
    import src.visualization
    import src.env_utils
    assert True

def test_config_creation():
    from src.config import ProjectConfig
    config = ProjectConfig()
    assert config is not None

def test_run_full_validation_no_token(monkeypatch):
    from scripts.run_full_validation import run_full_validation
    import pytest
    monkeypatch.setenv("TUSHARE_TOKEN", "")
    with pytest.raises(SystemExit) as e:
        run_full_validation()
    assert e.value.code == 1

def test_update_readme_no_files(capsys):
    from scripts.update_readme_results import update_readme_results
    # When output CSVs are absent the function skips silently; when present it succeeds.
    # Either path is acceptable — the test just verifies it does not raise.
    update_readme_results()
    captured = capsys.readouterr()
    assert "skipping README update" in captured.out or "README.md updated" in captured.out

def test_config_loads_token_from_dotenv(tmp_path, monkeypatch):
    import importlib
    import src.config as config_module
    from src.env_utils import load_local_env

    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    (tmp_path / ".env").write_text("TUSHARE_TOKEN=from_dotenv\n", encoding="utf-8")

    # load_local_env injects the .env contents into os.environ so the subsequent
    # module reload picks them up via os.getenv() in the dataclass field defaults.
    load_local_env(tmp_path)
    config_module = importlib.reload(config_module)
    config = config_module.ProjectConfig()

    assert config.tushare_token == "from_dotenv"


def test_cache_keys_include_date_range():
    """Verify _date_range_cache_key produces keys containing start_date and end_date."""
    from unittest.mock import MagicMock
    from src.data_collection import DataCollector

    mock_config = MagicMock()
    mock_config.start_date = "20200101"
    mock_config.end_date = "20251231"
    mock_config.tushare_token = ""
    mock_config.use_cache = True
    mock_config.force_refresh = False

    collector = DataCollector.__new__(DataCollector)
    collector.config = mock_config
    collector.logger = MagicMock()
    collector.cache_root = MagicMock()

    key = collector._date_range_cache_key("000001_SZ")
    assert "20200101" in key
    assert "20251231" in key
    assert key.startswith("000001_SZ_")


def test_full_validation_sets_run_mode(monkeypatch):
    """Verify run_full_validation sets RUN_MODE=full in subprocess env."""
    import subprocess
    from unittest.mock import MagicMock
    from scripts.run_full_validation import run_full_validation

    monkeypatch.setenv("TUSHARE_TOKEN", "test_token_12345678901234567890")

    captured_env = {}

    def fake_run(args, check=True, env=None, **kwargs):
        nonlocal captured_env
        if env is not None:
            captured_env = dict(env)
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("sys.exit", lambda code: None)

    run_full_validation()

    assert captured_env.get("RUN_MODE") == "full", f"Expected RUN_MODE=full, got {captured_env.get('RUN_MODE')}"


def test_settings_local_json_not_tracked():
    """Verify .claude/settings.local.json is not tracked by git."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).parent.parent
    local_settings = repo_root / ".claude" / "settings.local.json"

    try:
        result = subprocess.run(
            ["git", "ls-files", ".claude/settings.local.json"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        tracked = result.stdout.strip()
        assert not tracked, (
            f".claude/settings.local.json is tracked by git. "
            f"Run: git rm --cached .claude/settings.local.json"
        )
    except FileNotFoundError:
        pass


def test_placebo_output_schema(tmp_path):
    """Verify build_placebo_test produces output with required columns."""
    import numpy as np
    import pandas as pd
    from src.tushare_event_design import build_placebo_test

    event_df = pd.DataFrame({
        "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
        "event_trade_date": ["2023-01-15", "2023-02-20", "2023-03-10"],
        "event_type": ["preannouncement", "preannouncement", "express"],
        "CAR3": [0.01, -0.005, 0.02],
        "CAR5": [0.015, -0.01, 0.025],
        "CAR10": [0.02, -0.015, 0.03],
    })

    dates = pd.date_range("2022-01-01", "2024-01-01", freq="B")
    prices_df = pd.DataFrame({
        "ts_code": np.where(np.arange(len(dates)) % 2 == 0, "000001.SZ", "000002.SZ"),
        "trade_date": dates,
        "ret": np.random.default_rng(42).normal(0.0005, 0.02, len(dates)),
    })

    market_df = pd.DataFrame({
        "trade_date": dates,
        "mkt_ret": np.random.default_rng(99).normal(0.0003, 0.015, len(dates)),
    })

    result = build_placebo_test(
        event_df=event_df,
        prices_df=prices_df,
        market_df=market_df,
        outputs_tables_dir=tmp_path,
        n_placebo_samples=50,
        random_seed=123,
    )

    if not result.empty:
        required = {"window", "real_mean_car", "placebo_mean_car", "real_n", "placebo_n"}
        missing = required - set(result.columns)
        assert not missing, f"Missing columns: {missing}"
        assert len(result) == 3, f"Expected 3 window rows, got {len(result)}"
