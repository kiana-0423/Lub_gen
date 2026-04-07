from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from chemstudio.data.models import Base
from chemstudio.utils.paths import database_path

_ENGINES: dict[str, Engine] = {}
_SESSION_FACTORIES: dict[str, sessionmaker] = {}


def get_database_url(database_url: str | None = None) -> str:
    """解析最终数据库地址，优先显式参数，其次环境变量。"""
    if database_url:
        return database_url

    env_database_url = os.getenv("CHEMSTUDIO_DATABASE_URL")
    if env_database_url:
        return env_database_url

    return f"sqlite:///{database_path()}"


def _sqlite_path_from_url(database_url: str) -> Path | None:
    """从 SQLite URL 中提取本地文件路径，非文件型地址时返回空。"""
    sqlite_prefix = "sqlite:///"
    if not database_url.startswith(sqlite_prefix):
        return None
    raw_path = database_url[len(sqlite_prefix) :]
    if raw_path == ":memory:":
        return None
    return Path(raw_path)


def get_engine(database_url: str | None = None) -> Engine:
    """按数据库地址获取并缓存 SQLAlchemy Engine。"""
    resolved_url = get_database_url(database_url)
    engine = _ENGINES.get(resolved_url)
    if engine is None:
        connect_args = {"check_same_thread": False} if resolved_url.startswith("sqlite:///") else {}
        engine = create_engine(resolved_url, echo=False, future=True, connect_args=connect_args)
        _ENGINES[resolved_url] = engine
    return engine


def get_session_factory(database_url: str | None = None) -> sessionmaker:
    """按数据库地址获取并缓存 SQLAlchemy Session 工厂。"""
    resolved_url = get_database_url(database_url)
    factory = _SESSION_FACTORIES.get(resolved_url)
    if factory is None:
        factory = sessionmaker(
            bind=get_engine(resolved_url),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            class_=Session,
        )
        _SESSION_FACTORIES[resolved_url] = factory
    return factory


def initialize_database(database_url: str | None = None) -> Path | str:
    """初始化数据库表结构，并确保 SQLite 文件目录存在。"""
    resolved_url = get_database_url(database_url)
    sqlite_path = _sqlite_path_from_url(resolved_url)
    if sqlite_path is not None:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    Base.metadata.create_all(bind=get_engine(resolved_url))
    return sqlite_path or resolved_url


@contextmanager
def session_scope(database_url: str | None = None):
    """提供带自动提交和回滚处理的数据库会话上下文。"""
    session = get_session_factory(database_url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
