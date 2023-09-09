from typing import List

from sycamore.execution import Node, Rule
from sycamore.execution.rules import OptimizeResourceArgs, EnforceResourceUsage


class Rewriter:
    def __init__(self, extension_rules: List[Rule]):
        self.rules = [EnforceResourceUsage(), OptimizeResourceArgs(), *extension_rules]

    def rewrite(self, plan: Node) -> None:
        for rule in self.rules:
            plan.traverse_down(rule)
