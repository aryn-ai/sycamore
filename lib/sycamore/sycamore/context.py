from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from sycamore.llms import LLM
from sycamore.rules import Rule

if TYPE_CHECKING:
    import ray


class ExecMode(Enum):
    UNKNOWN = 0
    RAY = 1
    LOCAL = 2


@dataclass
class Context:
    """
    A class to implement a Sycamore Context, which initializes a Ray Worker and provides the ability
    to read data into a DocSet
    """

    exec_mode: ExecMode = ExecMode.RAY
    ray_args: Optional[dict[str, Any]] = None
    extension_rules: list[Rule] = field(default_factory=list)

    opensearch_client_config: Optional[dict[str, Any]] = None
    opensearch_index_name: Optional[str] = None
    opensearch_index_settings: Optional[dict[str, Any]] = None

    llm: Optional[LLM] = None

    @property
    def read(self):
        from sycamore.reader import DocSetReader

        return DocSetReader(self)

    def register_rule(self, rule: Rule) -> None:
        """
        Allows for the registration of Rules in the Sycamore Context that allow for communication with the
        underlying Ray context and can specify additional performance optimizations
        """
        self.extension_rules.append(rule)

    def get_extension_rule(self) -> list[Rule]:
        """
        Returns all Rules currently registered in the Context
        """
        return self.extension_rules

    def deregister_rule(self, rule: Rule) -> None:
        """
        Removes a currently registered Rule from the context
        """
        self.extension_rules.remove(rule)


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
    ray.shutdown()
