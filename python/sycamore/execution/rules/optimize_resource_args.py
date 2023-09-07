from sycamore.execution import (
    Node, NonCPUUser, NonGPUUser, Rule, SingleThreadUser)


class EnforceResourceUsage(Rule):
    def __call__(self, plan: Node) -> Node:
        if isinstance(plan, NonCPUUser):
            plan.resource_args["num_cpus"] = 0

        if isinstance(plan, SingleThreadUser):
            plan.resource_args["num_cpus"] = 1

        if isinstance(plan, NonGPUUser):
            plan.resource_args["num_gpus"] = 0
        return plan


class OptimizeResourceArgs(Rule):
    def __call__(self, plan: Node) -> Node:
        return plan
