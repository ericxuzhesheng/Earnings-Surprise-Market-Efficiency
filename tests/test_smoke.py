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
    assert True

def test_config_creation():
    from src.config import ProjectConfig
    config = ProjectConfig()
    assert config is not None

def test_run_full_validation_is_offline_by_default(monkeypatch):
    from scripts.run_full_validation import run_full_validation
    monkeypatch.setenv("TUSHARE_TOKEN", "")
    calls = []
    monkeypatch.setattr("scripts.run_full_validation.subprocess.run", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr("scripts.run_full_validation.Path.exists", lambda path: True)
    run_full_validation()
    assert len(calls) == 1

def test_update_readme_preserves_current_audit(monkeypatch, capsys):
    from scripts.update_readme_results import update_readme_results
    monkeypatch.setattr("scripts.update_readme_results.Path.exists", lambda path: True)
    update_readme_results()
    assert "refusing to restore" in capsys.readouterr().out
