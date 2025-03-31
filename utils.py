import logging
import os


def setup_logger(level=logging.DEBUG):
    logger = logging.getLogger("ts_guessing")
    logger.setLevel(level)

    log_file = os.path.join(os.path.dirname(__file__), "app.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
