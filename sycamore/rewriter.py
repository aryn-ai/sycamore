from sycamore.plan_nodes import Node
from sycamore.rules import Rule, OptimizeResourceArgs, EnforceResourceUsage


class Rewriter:
    def __init__(self, extension_rules: list[Rule]):
        self.rules = [EnforceResourceUsage(), OptimizeResourceArgs(), *extension_rules]

    def rewrite(self, plan: Node) -> None:
        for rule in self.rules:
            plan.traverse_down(rule)
