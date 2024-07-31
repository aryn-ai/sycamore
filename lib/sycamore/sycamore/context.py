from enum import Enum
import logging
import threading
from typing import Any, Optional, TYPE_CHECKING

from sycamore.config import Config
from sycamore.rules import Rule

if TYPE_CHECKING:
    import ray


def _ray_logging_setup():
    # The commented out lines allow for easier testing that logging is working correctly since
    # they will emit information at the start.

    # logging.error("RayLoggingSetup-Before (expect -After; if missing there is a bug)")

    ## WARNING: There can be weird interactions in jupyter/ray with auto-reload. Without the
    ## Spurious log [0-2]: messages below to verify that log messages are being properly
    ## propogated.  Spurious log 1 seems to somehow be required.  Without it, the remote map
    ## worker messages are less likely to come back.

    ## Some documentation for ray implies things should use the ray logger
    ray_logger = logging.getLogger("ray")
    ray_logger.setLevel(logging.INFO)
    # ray_logger.info("Spurious log 2: Verifying that log messages are propogated")

    ## Make the default logging show info messages
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Spurious log 1: Verifying that log messages are propogated")
    # logging.error("RayLoggingSetup-After-2Error")

    ## Verify that another logger would also log properly
    other_logger = logging.getLogger("other_logger")
    other_logger.setLevel(logging.INFO)
    # other_logger.info("RayLoggingSetup-After-3")


class ExecMode(Enum):
    UNKNOWN = 0
    RAY = 1
    LOCAL = 2


class Context:
    """
    A class to implement a Sycamore Context, which initializes a Ray Worker and provides the ability
    to read data into a DocSet
    """

    def __init__(
        self, exec_mode=ExecMode.RAY, ray_args: Optional[dict[str, Any]] = None, config: Optional[Config] = None
    ):
        self.exec_mode = exec_mode
        if self.exec_mode == ExecMode.RAY:
            import ray

            if ray_args is None:
                ray_args = {}

            if "logging_level" not in ray_args:
                ray_args.update({"logging_level": logging.INFO})

            if "runtime_env" not in ray_args:
                ray_args["runtime_env"] = {}

            if "worker_process_setup_hook" not in ray_args["runtime_env"]:
                # logging.error("Spurious log 0: If you do not see spurious log 1 & 2, log messages are being dropped")
                ray_args["runtime_env"]["worker_process_setup_hook"] = _ray_logging_setup

            ray.init(**ray_args)
        elif self.exec_mode == ExecMode.LOCAL:
            pass
        else:
            assert False, f"unsupported mode {self.exec_mode}"

        self.extension_rules: list[Rule] = []
        self._config = config or Config()
        self._internal_lock = threading.Lock()

    @property
    def read(self):
        from sycamore.reader import DocSetReader

        return DocSetReader(self)

    @property
    def config(self) -> Config:
        return self._config

    def register_rule(self, rule: Rule) -> None:
        """
        Allows for the registration of Rules in the Sycamore Context that allow for communication with the
        underlying Ray context and can specify additional performance optimizations
        """
        with self._internal_lock:
            self.extension_rules.append(rule)

    def get_extension_rule(self) -> list[Rule]:
        """
        Returns all Rules currently registered in the Context
        """
        with self._internal_lock:
            copied = self.extension_rules.copy()
        return copied

    def deregister_rule(self, rule: Rule) -> None:
        """
        Removes a currently registered Rule from the context
        """
        with self._internal_lock:
            self.extension_rules.remove(rule)


_context_lock = threading.Lock()
_global_context: Optional[Context] = None


def init(exec_mode=ExecMode.RAY, ray_args: Optional[dict[str, Any]] = None, config: Optional[Config] = None) -> Context:
    global _global_context
    with _context_lock:
        if _global_context is None:
            if ray_args is None:
                ray_args = {}

            # Set Logger for driver only, we consider worker_process_setup_hook
            # or runtime_env/config file for worker application log
            from sycamore.utils import sycamore_logger

            sycamore_logger.setup_logger()

            _global_context = Context(exec_mode, ray_args, config)

        return _global_context


def current(
    exec_mode=ExecMode.RAY, ray_args: Optional[dict[str, Any]] = None, config: Optional[Config] = None
) -> Context:
    if _global_context:
        return _global_context
    return init(exec_mode=exec_mode, ray_args=ray_args, config=config)


def shutdown() -> None:
    global _global_context
    with _context_lock:
        ray.shutdown()
        _global_context = None
