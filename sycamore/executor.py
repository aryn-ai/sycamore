from ray.data import Dataset

from sycamore import Context
from sycamore.plan_nodes import Node


class Execution:
    def __init__(self, context: Context, plan: Node):
        self._context = context
        self._plan = plan
        from sycamore.rewriter import Rewriter

        extension_rules = context.get_extension_rule()
        self.rewriter = Rewriter(extension_rules)

    def execute(self, plan: Node) -> "Dataset":
        plan_copied = plan.clone()
        self.rewriter.rewrite(plan_copied)
        return plan.execute()
