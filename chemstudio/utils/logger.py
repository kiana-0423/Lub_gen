from __future__ import annotations

import logging

from chemstudio.utils.config import AppConfig


def configure_logging() -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(AppConfig.LOG_DIR / "chemstudio.log", encoding="utf-8"),
        ],
    )
