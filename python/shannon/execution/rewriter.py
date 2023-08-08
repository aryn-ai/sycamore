from typing import List

from shannon.execution import (Node, Rule)
from shannon.execution.rules import OptimizeResourceArgs


class Rewriter:

    def __init__(self, extension_rules: List[Rule]):
        self.rules = [OptimizeResourceArgs(), *extension_rules]

    def rewrite(self, plan: Node) -> None:
        for rule in self.rules:
            rule(plan)
