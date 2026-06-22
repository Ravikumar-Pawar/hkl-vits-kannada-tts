import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Project root:
# hkl-vits-hybrid-kannada-tts/
BASE_DIR = Path(__file__).resolve().parents[2]

LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"


# create logs folder in root
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str = "app") -> logging.Logger:

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)


    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(filename)s:%(lineno)d | "
        "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


    # Console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)


    # File
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)


    logger.addHandler(console)
    logger.addHandler(file_handler)


    logger.propagate = False

    return logger