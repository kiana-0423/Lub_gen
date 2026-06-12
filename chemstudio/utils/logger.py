from __future__ import annotations

import logging
import os

from chemstudio.utils.config import AppConfig


def configure_logging() -> None:
    """Configure application-wide logging."""
    level_name = os.getenv("CHEMSTUDIO_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(AppConfig.LOG_DIR / "chemstudio.log", encoding="utf-8"),
        ],
    )
