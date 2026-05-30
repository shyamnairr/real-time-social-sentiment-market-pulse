"""
Loads config.yaml + .env and exposes a single `CONFIG` object plus helpers.

Usage:
    from utils.config import CONFIG, get_path
    mode = CONFIG["mode"]
    db_path = get_path("db_file")
"""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Project root = parent of this utils/ directory
ROOT = Path(__file__).resolve().parent.parent

# Load environment variables from .env (silent if missing -> demo mode still works)
load_dotenv(ROOT / ".env")


def _load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = _load_config()


def get_path(key: str) -> Path:
    """Resolve a path from config['paths'] to an absolute Path under ROOT."""
    rel = CONFIG["paths"][key]
    return (ROOT / rel).resolve()


def ensure_dirs() -> None:
    """Create all data directories if they don't exist (safe to call repeatedly)."""
    for key in ("data_dir", "raw_dir", "processed_dir", "models_dir"):
        get_path(key).mkdir(parents=True, exist_ok=True)


# --- Reddit credentials (live mode only) ----------------------------------
REDDIT_CREDS = {
    "client_id": os.getenv("REDDIT_CLIENT_ID"),
    "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    "user_agent": os.getenv("REDDIT_USER_AGENT", "market-pulse"),
}


def has_reddit_creds() -> bool:
    return bool(REDDIT_CREDS["client_id"]) and bool(REDDIT_CREDS["client_secret"])
