from sycamore.utils.time_trace import LogTime
import time


class TestTimeTrace:
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
