import logging
import time

from sycamore.utils.sycamore_logger import LoggerFilter


def test_logger_ratelimit(caplog):
    logger = logging.getLogger("test_sycamore")

    with caplog.at_level(logging.INFO):
        for i in range(5):
            logger.info(f"Unbounded {i}")

        logger.addFilter(LoggerFilter())
        for i in range(5):
            logger.info(f"Bounded {i}")

        time.sleep(1)
        logger.info("Bounded After")

    for i in range(5):
        assert f"Unbounded {i}\n" in caplog.text

    assert "Bounded 0" in caplog.text
    for i in range(1, 5):
        assert f"Bounded {i}\n" not in caplog.text

    assert "Bounded After\n" in caplog.text
