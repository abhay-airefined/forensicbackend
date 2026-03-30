"""Centralised logging configuration for the SFAS backend."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured console logging for the entire application."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("azure", "urllib3", "httpcore", "httpx", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    
    # Ensure verbose logging for our app modules
    logging.getLogger("app").setLevel(logging.DEBUG)
    logging.getLogger("sfas").setLevel(logging.DEBUG)

