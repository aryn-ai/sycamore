import threading
from typing import Optional

import ray

from shannon.execution import Execution, Rule


class Context:
    def __init__(self):
        # TODO, we need to handle what exactly the conf we pass down to Ray
        ray.init()
        self.execution = Execution()
        self._internal_lock = threading.Lock

    @property
    def read(self):
        from shannon.reader import DocSetReader
        return DocSetReader(self)

    def register(self, phase: str, rule: Rule) -> None:
        with self._internal_lock:
            self.execution.rules[phase].append(rule)

    def deregister(self, phase: str, rule: Rule) -> None:
        with self._internal_lock:
            self.execution.rules[phase].remove(rule)


_context_lock = threading.Lock()
_global_context: Optional[Context] = None


def init() -> Optional[Context]:
    global _global_context
    with _context_lock:
        if _global_context is None:
            _global_context = Context()

        return _global_context
