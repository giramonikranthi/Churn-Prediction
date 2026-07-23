import logging
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = APP_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"


def setup_logger() -> logging.Logger:
    """Create a reusable logger that writes to both console and file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("churn_app")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger()
