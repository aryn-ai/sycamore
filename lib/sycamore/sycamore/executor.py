import logging
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from ray.data import Dataset

from sycamore.context import Context, ExecMode
from sycamore.data import Document
from sycamore.plan_nodes import Node


def _ray_logging_setup():
    # The commented out lines allow for easier testing that logging is working correctly since
    # they will emit information at the start.

    # logging.error("RayLoggingSetup-Before (expect -After; if missing there is a bug)")

    ## WARNING: There can be weird interactions in jupyter/ray with auto-reload. Without the
    ## Spurious log [0-2]: messages below to verify that log messages are being properly
    ## propagated.  Spurious log 1 seems to somehow be required.  Without it, the remote map
    ## worker messages are less likely to come back.

    ## Some documentation for ray implies things should use the ray logger
    ray_logger = logging.getLogger("ray")
    ray_logger.setLevel(logging.INFO)
    # ray_logger.info("Spurious log 2: Verifying that log messages are propagated")

    ## Make the default logging show info messages
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Spurious log 1: Verifying that log messages are propagated")
    # logging.error("RayLoggingSetup-After-2Error")

    ## Verify that another logger would also log properly
    other_logger = logging.getLogger("other_logger")
    other_logger.setLevel(logging.INFO)
    # other_logger.info("RayLoggingSetup-After-3")


class Execution:
    def __init__(self, context: Context, plan: Node):
        self._context = context
        self._plan = plan
        self._exec_mode = context.exec_mode
        from sycamore.rewriter import Rewriter

        extension_rules = context.extension_rules
        self.rewriter = Rewriter(extension_rules)

    def execute(self, plan: Node, **kwargs) -> "Dataset":
        self.rewriter.rewrite(plan)
        if self._exec_mode == ExecMode.RAY:
            import ray

            if not ray.is_initialized():
                ray_args = self._context.ray_args or {}

                if "logging_level" not in ray_args:
                    ray_args.update({"logging_level": logging.INFO})

                if "runtime_env" not in ray_args:
                    ray_args["runtime_env"] = {}

                if "worker_process_setup_hook" not in ray_args["runtime_env"]:
                    # logging.error("Spurious log 0: If you do not see spurious log 1 & 2,
                    # log messages are being dropped")
                    ray_args["runtime_env"]["worker_process_setup_hook"] = _ray_logging_setup

                ray.init(**ray_args)
            return plan.execute(**kwargs)
        if self._exec_mode == ExecMode.LOCAL:
            from ray.data import from_items

            return from_items(items=[{"doc": doc.serialize()} for doc in self.recursive_execute(self._plan)])
        assert False, f"unsupported mode {self._exec_mode}"

    def execute_iter(self, plan: Node, **kwargs) -> Iterable[Document]:
        self.rewriter.rewrite(plan)
        if self._exec_mode == ExecMode.RAY:
            ds = plan.execute(**kwargs)
            for row in ds.iter_rows():
                yield Document.from_row(row)
            return
        if self._exec_mode == ExecMode.LOCAL:
            for d in self.recursive_execute(self._plan):
                yield d
            return
        assert False

    def recursive_execute(self, n: Node) -> list[Document]:
        if len(n.children) == 0:
            assert hasattr(n, "local_source"), f"Source {n} needs a local_source method"
            return n.local_source()
        if len(n.children) == 1:
            assert hasattr(n, "local_execute"), f"Transform {n.__class__.__name__} needs a local_execute method"
            assert n.children[0] is not None
            return n.local_execute(self.recursive_execute(n.children[0]))

        assert f"Unable to handle node {n} with multiple children"
        return []
