import logging.config
import os
from app.config.settings import settings

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "detailed": {
            "format": (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(pathname)s:%(lineno)d - %(message)s"
            ),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": settings.LOG_LEVEL,
            "formatter": "detailed",
            "filename": os.path.join(LOG_DIR, "omnibrain.log"),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
        },
    },
    "root": {
        "level": settings.LOG_LEVEL,
        "handlers": ["console", "file"],
    },
}


def setup_logging() -> None:
    """Configure logging using the dict defined above."""
    logging.config.dictConfig(LOGGING_CONFIG)