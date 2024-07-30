from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from ray.data import Dataset

from sycamore.context import Context, ExecMode
from sycamore.data import Document
from sycamore.plan_nodes import Node


class Execution:
    def __init__(self, context: Context, plan: Node):
        self._context = context
        self._plan = plan
        self._exec_mode = context.exec_mode
        from sycamore.rewriter import Rewriter

        extension_rules = context.get_extension_rule()
        self.rewriter = Rewriter(extension_rules)

    def execute(self, plan: Node, **kwargs) -> "Dataset":
        self.rewriter.rewrite(plan)
        if self._exec_mode == ExecMode.RAY:
            return plan.execute(**kwargs)
        if self._exec_mode == ExecMode.LOCAL:
            from ray.data import from_items

            return from_items(items=[{"doc": doc.serialize()} for doc in self.recursive_execute(self._plan)])
        assert False

    def execute_iter(self, plan: Node, **kwargs) -> Iterable[Document]:
        self.rewriter.rewrite(plan)
        if self._exec_mode == ExecMode.RAY:
            ds = plan.execute(**kwargs)
            for row in ds.iter_rows():
                yield Document.from_row(row)
            return
        if self._exec_mode == ExecMode.LOCAL:
            for d in self.recursive_execute(self._plan):
                yield d
            return
        assert False

    def recursive_execute(self, n: Node) -> list[Document]:
        if len(n.children) == 0:
            assert hasattr(n, "local_source"), f"Source {n} needs a local_source method"
            return n.local_source()
        if len(n.children) == 1:
            assert hasattr(n, "local_execute"), f"Transform {n.__class__.__name__} needs a local_execute method"
            assert n.children[0] is not None
            return n.local_execute(self.recursive_execute(n.children[0]))

        assert f"Unable to handle node {n} with multiple children"
        return []
