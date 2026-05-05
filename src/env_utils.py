from pathlib import Path


def load_local_env(project_root: Path) -> None:
    """Load environment variables from .env file in project_root if present.

    Uses python-dotenv with override=False so existing process env vars take
    precedence over file values (safe for CI environments that inject secrets).
    Silently does nothing if python-dotenv is not installed or .env is absent.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
