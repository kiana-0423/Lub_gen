from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def database_path() -> Path:
    return project_root() / "resources" / "chemstudio.sqlite"


def packaging_path() -> Path:
    return project_root() / "packaging"

