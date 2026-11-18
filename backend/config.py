import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def workspace_dir() -> Path:
    """Local storage root. Override with LOCALSIGHT_WORKSPACE (used by tests)."""
    ws = Path(os.environ.get("LOCALSIGHT_WORKSPACE", REPO_ROOT / "workspace"))
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def db_path() -> Path:
    return workspace_dir() / "localsight.db"


def frontend_dist_dir() -> Path:
    """Built frontend (npm run build). Served by the backend when present, for one-command launch."""
    return Path(os.environ.get("LOCALSIGHT_FRONTEND_DIST", REPO_ROOT / "frontend" / "dist"))
