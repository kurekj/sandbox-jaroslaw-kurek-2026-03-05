import sys

from loguru import logger

from src.v2.config import get_config


def init_loguru() -> None:
    """Initialize loguru logger."""
    logger_config = get_config().logger
    logger.remove()  # Remove the default logger
    logger.add(sys.stdout, level=logger_config.level)
