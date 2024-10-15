import logging
import sys
import time

from datetime import datetime, timedelta

handler_setup = False


def setup_logger():
    """Setup application logger"""
    global handler_setup
    if handler_setup:
        return
    handler_setup = True
    logger = logging.getLogger("sycamore")
    logger.setLevel(logging.INFO)

    logger_formatter = logging.Formatter("%(name)s - %(levelname)s - %(asctime)s - %(message)s")
    logger_formatter.converter = time.gmtime

    logger_handler = logging.StreamHandler(sys.stdout)
    logger_handler.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)


def get_logger():
    """Get an application logger"""
    logger = logging.getLogger("sycamore")
    return logger


class LoggerFilter(logging.Filter):
    def __init__(self, seconds=1):
        """
        A filter limit the rate of log messages.

        logger = logging.getLogger(__name__)
        logger.setFilter(LoggerFilter())


        Args:
            seconds: Minimum seconds between log messages.
        """

        self._min_interval = timedelta(seconds=seconds)
        self._next_log = datetime.now()

    def filter(self, record=None):
        if record is None:
            assert False
        now = datetime.now()
        if now >= self._next_log:
            self._next_log = now + self._min_interval
            return True

        return False
