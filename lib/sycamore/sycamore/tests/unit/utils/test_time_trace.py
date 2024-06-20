from sycamore.utils.time_trace import LogTime, TimeTrace
import time
import os
import tempfile

import pytest


class TestTimeTrace:
    @pytest.fixture(autouse=True)
    def set_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TIMETRACE"] = f"{tmp}/tt"
            yield

    def test_with(self):
        with TimeTrace("test_with"):
            time.sleep(0.01)

    def test_start_end(self):
        tt = TimeTrace("test_start_end")
        tt.start()
        time.sleep(0.01)
        tt.end()


class TestLogTime:
    def test_simple(self):
        a = LogTime("simple")
        a.start()
        time.sleep(1)
        d = a.measure()
        assert d.wall_s() >= 1
        assert d.user_s() <= 0.1
        assert d.sys_s() <= 0.1

    def test_point(self):
        # verify lack of crashing
        LogTime("point", point=True)

    def test_with_start(self):
        # verify lack of crashing
        with LogTime("start", log_start=True):
            pass
