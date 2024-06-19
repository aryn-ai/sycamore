import logging
import threading
from typing import Any, Optional

import ray
import ray.data

from sycamore.rules import Rule


class Context:
    def __init__(self, ray_args: Optional[dict[str, Any]] = None, enable_progress_bars=False):
        if ray_args is None:
            ray_args = {}

        if "logging_level" not in ray_args:
            ray_args.update({"logging_level": logging.WARNING})

        ray.init(**ray_args)

        # Disable progress bars by default while https://github.com/ray-project/ray/issues/44983 is open.
        ray.data.DataContext.get_current().enable_progress_bars = enable_progress_bars

        self.extension_rules: list[Rule] = []
        self._internal_lock = threading.Lock()

    @property
    def read(self):
        from sycamore.reader import DocSetReader

        return DocSetReader(self)

    def register_rule(self, rule: Rule) -> None:
        with self._internal_lock:
            self.extension_rules.append(rule)

    def get_extension_rule(self) -> list[Rule]:
        with self._internal_lock:
            copied = self.extension_rules.copy()
        return copied

    def deregister_rule(self, rule: Rule) -> None:
        with self._internal_lock:
            self.extension_rules.remove(rule)


_context_lock = threading.Lock()
_global_context: Optional[Context] = None


def init(ray_args: Optional[dict[str, Any]] = None, enable_progress_bars=False) -> Context:
    global _global_context
    with _context_lock:
        if _global_context is None:
            if ray_args is None:
                ray_args = {}

            # Set Logger for driver only, we consider worker_process_setup_hook
            # or runtime_env/config file for worker application log
            from sycamore.utils import sycamore_logger

            sycamore_logger.setup_logger()

            _global_context = Context(ray_args, enable_progress_bars=enable_progress_bars)

        return _global_context
