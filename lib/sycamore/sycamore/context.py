from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from sycamore.llms import LLM
from sycamore.rules import Rule


class ExecMode(Enum):
    UNKNOWN = 0
    RAY = 1
    LOCAL = 2


OS_CLIENT_ARGS = "client_args"
OS_INDEX_NAME = "index_name"
OS_INDEX_SETTINGS = "index_settings"


@dataclass
class Context:
    """
    A class to implement a Sycamore Context, which initializes a Ray Worker and provides the ability
    to read data into a DocSet
    """

    exec_mode: ExecMode = ExecMode.RAY
    ray_args: Optional[dict[str, Any]] = None

    """
    Allows for the registration of Rules in the Sycamore Context that allow for communication with the
    underlying Ray context and can specify additional performance optimizations
    """
    extension_rules: list[Rule] = field(default_factory=list)

    """
    Default OpenSearch configuration for a Context
    """
    opensearch_config: dict[str, Any] = field(default_factory=dict)

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
