import logging
import sys
import time

from shannon.context import (init, Context)
from shannon.docset import DocSet
from shannon.executor import Execution


logging.Formatter.converter = time.gmtime

loggers = logging.getLogger(__name__)
loggers.setLevel(logging.INFO)

logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(asctime)s - %(message)s')
logger_formatter.converter = time.gmtime

logger_handler = logging.StreamHandler(sys.stdout)
logger_handler.setLevel(logging.INFO)
logger_handler.setFormatter(logger_formatter)

loggers.addHandler(logger_handler)

__all__ = [
    "DocSet",
    "init",
    "Context",
    "Execution"
]
