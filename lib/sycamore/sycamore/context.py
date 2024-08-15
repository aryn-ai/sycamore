from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union
from sycamore.llms import LLM
from sycamore.plan_nodes import Node, NodeTraverse


class ExecMode(Enum):
    UNKNOWN = 0
    RAY = 1
    LOCAL = 2


@dataclass
class OpenSearchArgs:
    client_args: Optional[dict[str, Any]] = None
    index_name: Optional[str] = None
    index_settings: Optional[dict[str, Any]] = None


def _default_rewrite_rules():
    import sycamore.rules.optimize_resource_args as o

    return [o.EnforceResourceUsage(), o.OptimizeResourceArgs()]


@dataclass
class Context:
    """
    A class to implement a Sycamore Context, which initializes a Ray Worker and provides the ability
    to read data into a DocSet
    """

    exec_mode: ExecMode = ExecMode.RAY
    ray_args: Optional[dict[str, Any]] = None

    """
    Allows for the registration of Rules in the Sycamore Context that allow for transforming the
    nodes before execution.  These rules can optimize ray execution or perform other manipulations.
    """
    rewrite_rules: list[Union[Callable[[Node], Node], NodeTraverse]] = field(default_factory=_default_rewrite_rules)

    """
    Default OpenSearch args for a Context
    """
    opensearch_args: Optional[OpenSearchArgs] = None

    """
    Default LLM for a Context
    """
    llm: Optional[LLM] = None

    @property
    def read(self):
        from sycamore.reader import DocSetReader

        return DocSetReader(self)


def init(exec_mode=ExecMode.RAY, ray_args: Optional[dict[str, Any]] = None, **kwargs) -> Context:
    """
    Initialize a new Context.
    """
    if ray_args is None:
        ray_args = {}

    # Set Logger for driver only, we consider worker_process_setup_hook
    # or runtime_env/config file for worker application log
    from sycamore.utils import sycamore_logger

    sycamore_logger.setup_logger()

    return Context(exec_mode=exec_mode, ray_args=ray_args, **kwargs)


def shutdown() -> None:
    import ray

    ray.shutdown()
