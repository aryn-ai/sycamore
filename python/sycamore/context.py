import threading
from typing import (Any, Dict, List, Optional)

import ray

from sycamore.execution import Rule


class Context:
    def __init__(self, ray_args: Dict[str, Any] = None):
        ray.init(**ray_args)
        self.extension_rules: List[Rule] = []
        self._internal_lock = threading.Lock()

    @property
    def read(self):
        from sycamore.reader import DocSetReader
        return DocSetReader(self)

    def register_rule(self, rule: Rule) -> None:
        with self._internal_lock:
            self.extension_rules.append(rule)

    def get_extension_rule(self) -> List[Rule]:
        with self._internal_lock:
            copied = self.extension_rules.copy()
        return copied

    def deregister_rule(self, rule: Rule) -> None:
        with self._internal_lock:
            self.extension_rules.remove(rule)


_context_lock = threading.Lock()
_global_context: Optional[Context] = None


def init(ray_args: Dict[str, Any] = None) -> Optional[Context]:
    global _global_context
    with _context_lock:
        if _global_context is None:
            if ray_args is None:
                ray_args = {}

            # Set Logger for driver only, we consider worker_process_setup_hook
            # or runtime_env/config file for worker application log
            from sycamore.utils import sycamore_logger
            sycamore_logger.setup_logger()

            _global_context = Context(ray_args)

        return _global_context
