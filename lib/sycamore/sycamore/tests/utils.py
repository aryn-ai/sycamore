import logging
import sys


def setup_debug_logging(module_name: str):
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    # Create a StreamHandler that writes to sys.stdout
    handler = logging.StreamHandler(sys.stdout)
    # Set the desired logging level for the handler (optional, but good practice)
    handler.setLevel(logging.INFO)  # Only INFO level and above will be handled by this handler
    # Create a formatter for the handler to define the log message format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(handler)
