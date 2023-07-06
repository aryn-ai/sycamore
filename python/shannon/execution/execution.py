from ray.data import Dataset
from shannon.execution import (Node, Rule)
from shannon.execution.rules import (
    PushEmbeddingModelConstraint, OptimizeResourceArgs)
from typing import (Dict, List)


class Execution:

    def __init__(self):
        self.rules: Dict[str, List[Rule]] = {
            "rewrite": [PushEmbeddingModelConstraint(), OptimizeResourceArgs()]
        }

    def rewrite(self, plan: Node) -> None:
        rewrite_rules = self.rules["rewrite"]
        for rule in rewrite_rules:
            rule(plan)

    def execute(self, plan: Node) -> "Dataset":
        self.rewrite(plan)
        return plan.execute()
