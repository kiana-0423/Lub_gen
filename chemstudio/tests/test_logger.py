from __future__ import annotations

import logging

from chemstudio.utils import logger as logger_module


def test_configure_logging_uses_environment_level(monkeypatch):
    captured: dict[str, object] = {}

    def fake_basic_config(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setenv("CHEMSTUDIO_LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(logger_module.logging, "basicConfig", fake_basic_config)

    logger_module.configure_logging()

    assert captured["level"] == logging.DEBUG


def test_configure_logging_falls_back_to_info_for_invalid_level(monkeypatch):
    captured: dict[str, object] = {}

    def fake_basic_config(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setenv("CHEMSTUDIO_LOG_LEVEL", "NOT_A_LEVEL")
    monkeypatch.setattr(logger_module.logging, "basicConfig", fake_basic_config)

    logger_module.configure_logging()

    assert captured["level"] == logging.INFO
