import threading
from typing import (List, Optional)

import ray

from sycamore.execution import Rule


class Context:
    def __init__(self):
        # TODO, we need to handle what exactly the conf we pass down to Ray
        ray.init()
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


def init() -> Optional[Context]:
    global _global_context
    with _context_lock:
        if _global_context is None:
            _global_context = Context()

        return _global_context
