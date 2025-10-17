"""pytest configuration and fixtures for the projector_mapper test suite."""

from pathlib import Path


def pytest_configure(config) -> None:
    """Hook called after command line options have been parsed."""
    load_env()


def load_env() -> None:
    """Load .env file from project root or parent directories."""
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Search up to 5 levels
        env_file = current / ".env"
        if env_file.exists():
            _parse_env_file(env_file)
            return
        current = current.parent


def _parse_env_file(path: Path) -> None:
    """Parse .env file and set environment variables."""
    import os
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and not os.getenv(key):  # Don't override existing env vars
                    os.environ[key] = value
