import functools
import uuid
from queue import Queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union, List
import inspect

import ray

from sycamore.plan_nodes import Node, NodeTraverse


class ExecMode(Enum):
    UNKNOWN = 0
    RAY = 1
    LOCAL = 2


class OperationTypes(Enum):
    DEFAULT = "default"
    BINARY_CLASSIFIER = "binary_classifier"
    INFORMATION_EXTRACTOR = "information_extractor"
    TEXT_SIMILARITY = "text_similarity"


def _default_rewrite_rules():
    import sycamore.rules.optimize_resource_args as o

    return [o.EnforceResourceUsage(), o.OptimizeResourceArgs()]


class State:
    """
    Storage for context level data. This can be used to store things like metrics, failed records, etc.
    """

    def __init__(self):
        self.records: Queue = Queue()

    def append(self, entry: dict[str, Any]):
        self.records.put(entry)

    def get(self) -> Optional[list[dict[str, Any]]]:
        return list(self.records.queue)


class StateManager:
    """
    Interface to interact with Context level state. Depending on how the state is persistent (local vs distributed etc.)
    the implementation of this class will vary.
    """

    def append(self, entry: dict[str, Any]):
        raise NotImplementedError

    def get(self) -> Optional[list[dict[str, Any]]]:
        raise NotImplementedError


@ray.remote
class RayStateActor:
    """
    A proxy to Context state that can be used in a Ray distributed environment.
    """

    def __init__(self):
        self.local_state = State()  # Utilize a local GlobalState for internal operations

    def append(self, entry: dict[str, Any]):
        self.local_state.append(entry)

    def get(self) -> Optional[list[dict[str, Any]]]:
        return self.local_state.get()


class RayGlobalStateManager(StateManager):
    """
    Ray state manager implementation. Interacts with a remote RayStateActor to support distributed state.
    """

    def __init__(self):
        self.ray_state = RayStateActor.remote()

    def append(self, entry: dict[str, Any]):
        self.ray_state.append.remote(entry)

    def get(self) -> Optional[list[dict[str, Any]]]:
        return ray.get(self.ray_state.get.remote())


class InMemoryStateManager(StateManager):
    """
    In-memory context state management.
    """

    def __init__(self):
        self.local_state = State()

    def append(self, entry: dict[str, Any]):
        self.local_state.append(entry)

    def get(self) -> Optional[list[dict[str, Any]]]:
        return self.local_state.get()


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
    params: dict[str, Any] = field(default_factory=dict)

    state: Optional[StateManager] = None

    context_id: str = str(uuid.uuid4())

    def add_state_data(self, entry: dict[str, Any]):
        if self.state:
            entry["context_id"] = self.context_id
            self.state.append(entry)

    def get_state_data(self) -> Optional[list[Any]]:
        if self.state:
            return self.state.get()
        return None

    @property
    def read(self):
        from sycamore.reader import DocSetReader

        return DocSetReader(self)


def get_val_from_context(
    context: "Context", val_key: str, param_names: Optional[List[str]] = None, ignore_default: bool = False
) -> Optional[Any]:
    """
    Helper function: Given a Context object, return the possible value for a given val.
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

    if param_names:
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

    There is a fair bit of complexity regarding arg management but the comments should be clear.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0] if len(args) > 0 else {}
            ctx = kwargs.get("context", getattr(self, "context", getattr(self, "_context", None)))
            if ctx and ctx.params:

                """
                Create argument candidates 'candidate_kwargs' from the Context
                """
                candidate_kwargs = {}
                candidate_kwargs.update(ctx.params.get("default", {}))
                qual_name = func.__qualname__  # e.g. 'DocSetWriter.opensearch'
                function_name_wo_class = qual_name.split(".")[-1]  # e.g. 'opensearch'
                candidate_kwargs.update(ctx.params.get(function_name_wo_class, {}))
                candidate_kwargs.update(ctx.params.get(qual_name, {}))
                for i in names:
                    candidate_kwargs.update(ctx.params.get(i, {}))

                """
                If positional args are provided, we want to pop those keys from candidate_kwargs
                """
                sig = inspect.signature(func)
                signature = list(sig.parameters.keys())
                for param in signature[: len(args)]:
                    candidate_kwargs.pop(param, None)

                """
                If the function doesn't accept arbitrary kwargs, we don't want to use candidate_kwargs that aren't in 
                the function signature.
                """
                new_kwargs = {}
                accepts_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

                if accepts_kwargs:
                    new_kwargs = candidate_kwargs
                else:
                    for param in signature[len(args) :]:  # arguments that haven't been passed as positional args
                        candidate_val = candidate_kwargs.get(param)
                        if candidate_val:
                            new_kwargs[param] = candidate_val

                """
                Put in user provided kwargs (either through decorator param or function call)
                """
                new_kwargs.update(kwargs)

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
