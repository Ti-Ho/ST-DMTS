import logging
import os
import sys


def get_logger(log_path="log", log_file="log.txt"):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    log_file = os.path.join(log_path, log_file)

    # file log
    file_hander = logging.FileHandler(log_file, mode='w')
    file_hander.setFormatter(logging.Formatter('%(levelname)s -> %(asctime)s: %(message)s'))
    file_hander.setLevel(logging.INFO)

    # console log
    console_hander = logging.StreamHandler(sys.stdout)
    console_hander.setFormatter(logging.Formatter('%(levelname)s -> %(asctime)s: %(message)s'))
    console_hander.setLevel(logging.INFO)

    # Configure
    logging.basicConfig(level=min(file_hander.level, console_hander.level), handlers=[file_hander, console_hander])
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.info("test2")
