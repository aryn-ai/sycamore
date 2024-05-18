from sycamore.plan_nodes import Node, NonCPUUser, NonGPUUser, SingleThreadUser


class Rule:
    def __call__(self, plan: Node) -> Node:
        raise NotImplementedError


class EnforceResourceUsage(Rule):
    def __call__(self, plan: Node) -> Node:
        if isinstance(plan, NonCPUUser):
            plan.resource_args["num_cpus"] = 0

        if isinstance(plan, SingleThreadUser) and "num_cpus" not in plan.resource_args:
            plan.resource_args["num_cpus"] = 1

        if isinstance(plan, NonGPUUser):
            assert "num_gpus" not in plan.resource_args

        return plan


class OptimizeResourceArgs(Rule):
    def __call__(self, plan: Node) -> Node:
        return plan
