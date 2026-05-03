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

def test_run_full_validation_no_token(monkeypatch):
    from scripts.run_full_validation import run_full_validation
    import pytest
    monkeypatch.setenv("TUSHARE_TOKEN", "")
    with pytest.raises(SystemExit) as e:
        run_full_validation()
    assert e.value.code == 1

def test_update_readme_no_files(tmp_path, monkeypatch):
    from scripts.update_readme_results import update_readme_results
    # Mock project root to a temp directory
    monkeypatch.setattr("scripts.update_readme_results.Path.parent", tmp_path)
    # Should exit silently if files missing
    update_readme_results()
    assert True
