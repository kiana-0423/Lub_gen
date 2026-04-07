from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """返回项目根目录路径。"""
    return Path(__file__).resolve().parents[2]


def package_root() -> Path:
    """返回 `chemstudio` 包目录路径。"""
    return project_root() / "chemstudio"


def database_path() -> Path:
    """返回默认数据库文件路径，允许通过环境变量覆盖。"""
    configured = os.getenv("CHEMSTUDIO_DATABASE_PATH")
    if configured:
        return Path(configured)
    return package_root() / "resources" / "chemstudio.sqlite"


def model_store_path() -> Path:
    """返回模型文件存储目录，允许通过环境变量覆盖。"""
    configured = os.getenv("CHEMSTUDIO_MODEL_STORE_PATH")
    if configured:
        return Path(configured)
    return package_root() / "resources" / "models"
