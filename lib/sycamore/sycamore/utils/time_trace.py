import os
import time
import struct
import resource
import threading
import functools


class TimeTrace:
    fd = -1

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
        self.t0 = time.time_ns()
        self.r0 = resource.getrusage(resource.RUSAGE_THREAD)

    def end(self):
        if TimeTrace.fd < 0:
            return
        t1 = time.time_ns()
        r1 = resource.getrusage(resource.RUSAGE_THREAD)
        thr = threading.get_native_id()
        r0 = self.r0
        user = int((r1.ru_utime - r0.ru_utime) * 1000000000.0)
        syst = int((r1.ru_stime - r0.ru_stime) * 1000000000.0)
        rss = r1.ru_maxrss * 1024  # could do max, but this is more granular
        buf = struct.pack(
            "BxxxIQQQQQ48s",
            0,  # version
            thr,
            self.t0,
            t1,
            user,
            syst,
            rss,
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
