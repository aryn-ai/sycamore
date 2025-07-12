import datetime
import time


def sleep_until(unix_timestamp):
    duration = unix_timestamp - time.time()
    if duration > 0:
        time.sleep(duration)
    return f"Slept {duration}s until " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
