import os
import time
import struct
import resource
import threading
import functools
import logging
from sys import platform

zero_time = time.time_ns()
# Mac reports ru_maxrss in bytes. Linux reports it in kB.
if platform == "darwin":
    RSS_MULTIPLIER = 1
else:
    RSS_MULTIPLIER = 1024


logger = logging.getLogger(__name__)


class TimeTraceData:
    def __init__(self, t0, t1, user, sys, rss):
        self.t0 = t0  # ns
        self.t1 = t1  # ns
        self.user = user  # ns
        self.sys = sys  # ns
        self.rss = rss  # bytes

    def wall_s(self):
        return (self.t1 - self.t0) / 1.0e9

    def user_s(self):
        return self.user / 1.0e9

    def sys_s(self):
        return self.sys / 1.0e9

    def rss_mib(self):
        return self.rss / (1024 * 1024)


class InMemoryTimeTrace:
    try:
        resource_type = resource.RUSAGE_THREAD  # type: ignore
    except AttributeError:
        resource_type = resource.RUSAGE_SELF

    def __init__(self):
        self.t0 = time.time_ns()
        self.r0 = resource.getrusage(self.resource_type)

    def measure(self):
        t1 = time.time_ns()
        r1 = resource.getrusage(self.resource_type)
        r0 = self.r0
        user = int((r1.ru_utime - r0.ru_utime) * 1.0e9)
        sys = int((r1.ru_stime - r0.ru_stime) * 1.0e9)

        rss = r1.ru_maxrss * RSS_MULTIPLIER  # could do max(r0,r1), but this is more specific

        return TimeTraceData(self.t0, t1, user, sys, rss)


class TimeTrace:
    fd = -1

    try:
        resource_type = resource.RUSAGE_THREAD  # type: ignore
    except AttributeError:
        resource_type = resource.RUSAGE_SELF

    def __init__(self, name: str):
        self.name = name.encode()
        self._setup()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def start(self):
        if TimeTrace.fd < 0:
            return
        self.imtt = InMemoryTimeTrace()

    def end(self):
        if TimeTrace.fd < 0:
            return
        data = self.imtt.measure()
        thr = threading.get_native_id()

        buf = struct.pack(
            "BxxxIQQQQQ48s",
            0,  # version
            thr,
            data.t0,
            data.t1,
            data.user,
            data.sys,
            data.rss,
            self.name,
        )
        os.write(TimeTrace.fd, buf)

    @classmethod
    def _setup(cls):
        if cls.fd >= 0:
            return
        pfx = os.environ.get("TIMETRACE")
        if not pfx:
            return
        ts = time.strftime("%Y%m%d%H%M%S")
        pid = os.getpid()
        path = f"{pfx}.{ts}.{pid}"
        cls.fd = os.open(  # ??? when do we close?
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            mode=0o644,
        )


def timetrace(name: str):
    """
    Decorator for TimeTrace.  Use like this:

    @timetrace("label")
    def foo():
        time.sleep(1.0)
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with TimeTrace(name):
                return f(*args, **kwargs)

        return wrapper

    return decorator


class _ZeroRU:
    def __init__(self):
        self.ru_utime = 0
        self.ru_stime = 0


class LogTime:
    def __init__(self, name: str, *, point: bool = False, log_start: bool = False):
        if point:
            self._logpoint(name)
            return

        self.name = name
        self.log_start = log_start

    def __enter__(self):
        self.start()

    def start(self):
        self.imtt = InMemoryTimeTrace()
        if self.log_start:
            self._logpoint(self.name + "_start")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.measure()

    def measure(self):
        d = self.imtt.measure()
        self._log(self.name, d)
        return d

    def _logpoint(self, name):
        t = InMemoryTimeTrace()
        t.t0 = zero_time
        t.r0 = _ZeroRU()
        self._log(name, t.measure())

    def _log(self, name, d):
        logger.info(
            f"{name} wall: {d.wall_s():6.3f} user: {d.user_s():6.3f}sys: {d.sys_s():6.3f} rss_mib: {d.rss_mib():4.3f}"
        )
