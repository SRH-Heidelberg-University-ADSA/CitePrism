# src/utils/logger.py
import logging
import sys
from logging.handlers import RotatingFileHandler
import os
import yaml


def setup_logger(config_path="config/config.yaml"):
    """
    Sets up a Unicode-safe logger based on the configuration file.
    """

    # Load minimal config just for logging
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    filename = log_config.get('log_file', 'extractor.log')

    # Create logger
    logger = logging.getLogger("ResearchPaperExtractor")
    logger.setLevel(level)
    logger.propagate = False  # Prevent double logging

    # Clear existing handlers (important for re-runs)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # -----------------------------
    # Console Handler (UTF-8 SAFE)
    # -----------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # 🔑 CRITICAL FIX: force UTF-8 with fallback
    try:
        console_handler.stream.reconfigure(
            encoding="utf-8",
            errors="replace"
        )
    except Exception:
        # Fallback for older Python versions
        pass

    logger.addHandler(console_handler)

    # -----------------------------
    # File Handler (UTF-8 SAFE)
    # -----------------------------
    file_handler = RotatingFileHandler(
        filename,
        maxBytes=log_config.get('max_log_size_mb', 10) * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",          # 🔑 FIX
        errors="replace"           # 🔑 FIX
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Singleton instance
logger = setup_logger()
