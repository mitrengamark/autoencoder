"""Load project .env and expose Hugging Face credentials to child processes."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


def load_project_env(root: Path | None = None) -> None:
    """Load .env from project root and map HF_TOKEN for Hugging Face clients."""
    env_root = root or PROJECT_ROOT
    load_dotenv(env_root / ".env")

    token = os.environ.get("HF_TOKEN")
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
