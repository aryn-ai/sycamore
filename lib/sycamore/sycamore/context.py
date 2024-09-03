import functools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union, List

from sycamore.plan_nodes import Node, NodeTraverse


class ExecMode(Enum):
    UNKNOWN = 0
    RAY = 1
    LOCAL = 2


class OperationTypes(Enum):
    DEFAULT = "default"
    BINARY_CLASSIFIER = "binary_classifier"
    INFORMATION_EXTRACTOR = "information_extractor"


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
    Define parameters for global usage
    """
    params: Optional[dict[str, Any]] = None

    @property
    def read(self):
        from sycamore.reader import DocSetReader

        return DocSetReader(self)


def get_val_from_context(
    context: "Context", val_key: str, param_names: List[str], ignore_default: bool = False
) -> Optional[Any]:
    """
    GIven a Context object, return the possible value for a given val.
    This assumes context.params is not a nested dict.
    @param context: Context to use
    @param val_key: Key for the value to be returned
    @param param_names: List of parameter namespaces to look for.
        Always uses OperationTypes.DEFAULT unless configured otherwise.
    @param ignore_default: disable usage for OperationTypes.DEFAULT parameter namespace
    @return: Optional value given configs.
    """
    if not context.params:
        return None

    for param_name in param_names:
        cand = context.params.get(param_name, {}).get(val_key)
        if cand is not None:
            return cand

    if not ignore_default:
        return context.params.get(OperationTypes.DEFAULT.value, {}).get(val_key)

    return None


def context_params(*names):
    """
    Applies kwargs from the context to a function call. Requires 'context': Context, to be an argument to the method.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0] if len(args) > 0 else {}
            ctx = kwargs.get("context", getattr(self, "context", getattr(self, "_context", None)))
            if ctx and ctx.params:
                new_kwargs = {}
                new_kwargs.update(ctx.params.get("default", {}))
                qual_name = func.__qualname__  # e.g. 'DocSetWriter.opensearch'
                function_name_wo_class = qual_name.split(".")[-1]
                new_kwargs.update(ctx.params.get(function_name_wo_class, {}))
                new_kwargs.update(ctx.params.get(qual_name, {}))

                for i in names:
                    new_kwargs.update(ctx.params.get(i, {}))
                new_kwargs.update(kwargs)

                """
                If positional args are provided, we want to pop those keys from new_kwargs that have been deduced 
                from context
                """
                signature = func.__code__.co_varnames[: func.__code__.co_argcount]
                for param in signature[: len(args)]:
                    if param == "self":
                        continue
                    new_kwargs.pop(param)

                return func(*args, **new_kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    """
        this let's you handle decorator usage like:
        @context_params OR
        @context_params() OR 
        @context_params("template") OR 
        @context_params("template1", "template2")
    """
    if len(names) == 1 and callable(names[0]):
        return decorator(names[0])
    else:
        return decorator


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
