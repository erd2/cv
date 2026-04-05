"""Логирование доминирующей эмоции в файл."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_emotion_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("face_match.emotion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(fh)
    return logger


def log_emotion(logger: logging.Logger, label: str, similarity: float, match_name: str) -> None:
    logger.info("emotion=%s | similarity=%.3f | match=%s", label, similarity, match_name)
