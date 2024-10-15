import logging
from typing import Any, Callable, Iterable, Optional, Union, TYPE_CHECKING

import numpy as np

from sycamore.data import Document, MetadataDocument
from sycamore.utils.lineage_utils import update_lineage
from sycamore.data.document import split_data_metadata
from sycamore.plan_nodes import Node, UnaryNode
from sycamore.utils.ray_utils import check_serializable

if TYPE_CHECKING:
    from ray.data import Dataset, Datasink


def take_separate(dataset: "Dataset", limit: Optional[int] = None) -> tuple[list[Document], list[MetadataDocument]]:
    """
    Returns the list of documents from a dataset separating out data and metadata docs.
    """
    if limit is None:
        raw = dataset.take_all()
    else:
        raw = dataset.take(limit)

    all = [Document.from_row(d) for d in raw]
    return split_data_metadata(all)


# Once we do python 3.12+ only, this can be:
# def _noneOr[T](a: T, default: T) -> T:
def _noneOr(a: Any, default: Any) -> Any:
    if a is None:
        return default
    return a


def rename(new_function_name: str):
    def decorator(f):
        f.__name__ = new_function_name
        return f

    return decorator


def get_name_from_callable(f):
    # check this condition first. A type will have a __name__ but might not have __call__
    if isinstance(f, type):
        if "__call__" in dir(f):
            return f.__name__
        else:
            raise ValueError("f argument is a class without an __call__ method")

    # Can't do "__name__" in dir(f), dir(f) doesn't always list __name__, so try to use it and fail
    try:
        return f.__name__
    except AttributeError:
        pass

    if "__call__" in dir(f):
        return f.__class__.__name__
    else:
        raise ValueError("f argument is an object without an __call__ method")

    raise ValueError(f"Unable to extract name from {f}, dir(f): {dir(f)}")


class BaseMapTransform(UnaryNode):
    """
    BaseMapTransform abstracts away MetadataDocuments from all other transforms.

    If f is a class type, the class will be instantiated and run as an actor in ray.
    The parallelism will default to 1 if unspecified. constructor_args and
    constructor_kwargs can be used to provide arguments when initializing the class

    If f is an object type and parallelism is specified, it will run as an actor
    Otherwise f will be run as a function.

    Use args, kwargs to pass additional args to the function call.
    """

    def __init__(
        self,
        child: Optional[Node],
        *,
        f: Any,  # Callable(doc, args, kwargs) or Class(c_args, c_kwargs).__call__(doc, args, kwargs)
        name: Optional[str] = None,
        args: Optional[Iterable[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        constructor_args: Optional[Iterable[Any]] = None,
        constructor_kwargs: Optional[dict[str, Any]] = None,
        # If we auto-generate lineage, then the conversion to BaseMap has to go in a single PR
        # since everything needs to be updated to skip metadata. If we temporarily disable the
        # lineage metadata, then we can do the conversion to BaseMap in separate PRs.
        enable_auto_metadata: bool = True,
        **resource_args,
    ):
        if isinstance(f, type) and "parallelism" not in resource_args:
            resource_args["parallelism"] = 1

        super().__init__(child, **resource_args)
        if name is None:
            name = get_name_from_callable(f)

        self._f = f
        self._name = name
        self._args = args
        self._kwargs = kwargs
        self._constructor_args = constructor_args
        self._constructor_kwargs = constructor_kwargs
        self._enable_auto_metadata = enable_auto_metadata

    def execute(
        self,
        write_intermediate_data: bool = False,
        intermediate_datasink: Optional[Union[type["Datasink"], "Datasink"]] = None,
        intermediate_datasink_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> "Dataset":
        # If serializability fails, the error messages are very confusing. These checks
        # give a much more sensible error message and give it before ray starts execution.
        check_serializable(
            self._f, self._name, self._args, self._kwargs, self._constructor_args, self._constructor_kwargs
        )

        from ray.data import ActorPoolStrategy

        if "num_gpus" in self.resource_args:
            assert self.resource_args["num_gpus"] > 0

        input_dataset = self.child().execute(
            write_intermediate_data=write_intermediate_data,
            intermediate_datasink=intermediate_datasink,
            intermediate_datasink_kwargs=intermediate_datasink_kwargs,
        )
        if isinstance(self._f, type):  # is f a class?
            # Maybe add a class as function variant if the caller specified parallelism=None
            result = input_dataset.map_batches(self._map_class(), **self.resource_args)
        elif "compute" in self.resource_args:
            assert isinstance(
                self.resource_args["compute"], ActorPoolStrategy
            ), "only supported compute type is ActorPoolStrategy"
            # Ray requires a class for ActorPoolStrategy
            result = input_dataset.map_batches(self._map_callable_as_class(), **self.resource_args)
        else:
            result = input_dataset.map_batches(self._map_function(), **self.resource_args)

        if write_intermediate_data:
            assert intermediate_datasink is not None
            if isinstance(intermediate_datasink, type):
                assert intermediate_datasink_kwargs is not None
                # ensure each nodes data is written in a separate directory
                path = intermediate_datasink_kwargs["path"] + "/" + self._name
                intermediate_datasink = intermediate_datasink(**{**intermediate_datasink_kwargs, "path": path})
            else:
                intermediate_datasink = intermediate_datasink
            result.write_datasink(intermediate_datasink)
        return result

    def local_execute(self, all_docs: list[Document]) -> list[Document]:
        docs = [d for d in all_docs if not isinstance(d, MetadataDocument)]
        metadata = [d for d in all_docs if isinstance(d, MetadataDocument)]
        outputs = self._local_process(docs)
        to_docs = [d for d in outputs if not isinstance(d, MetadataDocument)]
        if self._enable_auto_metadata and (len(docs) > 0 or len(to_docs) > 0):
            outputs.extend(update_lineage(docs, to_docs))
        outputs.extend(metadata)
        return outputs

    def _local_process(self, in_docs: list[Document]) -> list[Document]:
        """Internal function for faster testing during the conversion to running on BaseMap.
        If extended with metadata support, this could become more real."""
        import copy

        # transforms assume they can mutate docs in place; this works in ray because documents are serialized and
        # deserialized between every stage.
        docs = copy.deepcopy(in_docs)
        if isinstance(self._f, type):  # is f a class?
            c_args = _noneOr(self._constructor_args, tuple())
            c_kwargs = _noneOr(self._constructor_kwargs, {})
            inst = self._f(*c_args, **c_kwargs)

            args = _noneOr(self._args, tuple())
            kwargs = _noneOr(self._kwargs, {})
            return inst(docs, *args, **kwargs)
        else:
            args = _noneOr(self._args, tuple())
            kwargs = _noneOr(self._kwargs, {})
            return self._f(docs, *args, **kwargs)

    def _map_function(self):
        f = self._f
        name = self._name
        args = _noneOr(self._args, tuple())
        kwargs = _noneOr(self._kwargs, {})
        enable_auto_metadata = self._enable_auto_metadata

        @rename(name)
        def ray_callable(ray_input: dict[str, np.ndarray]) -> dict[str, list]:
            return BaseMapTransform._process_ray(ray_input, name, lambda d: f(d, *args, **kwargs), enable_auto_metadata)

        return ray_callable

    def _map_callable_as_class(self):
        f = self._f
        name = self._name
        args = _noneOr(self._args, tuple())
        kwargs = _noneOr(self._kwargs, {})
        enable_auto_metadata = self._enable_auto_metadata

        def ray_init(self):
            pass

        def ray_callable(self, ray_input: dict[str, np.ndarray]) -> dict[str, list]:
            return BaseMapTransform._process_ray(ray_input, name, lambda d: f(d, *args, **kwargs), enable_auto_metadata)

        return type("BaseMapTransformCallable__" + name, (), {"__init__": ray_init, "__call__": ray_callable})

    def _map_class(self):
        c = self._f
        name = self._name
        args = _noneOr(self._args, tuple())
        kwargs = _noneOr(self._kwargs, {})
        c_args = _noneOr(self._constructor_args, tuple())
        c_kwargs = _noneOr(self._constructor_kwargs, {})
        enable_auto_metadata = self._enable_auto_metadata

        def ray_init(self):
            self.base = c(*c_args, **c_kwargs)

        def ray_callable(self, ray_input: dict[str, np.ndarray]) -> dict[str, list]:
            return BaseMapTransform._process_ray(
                ray_input, name, lambda d: self.base(d, *args, **kwargs), enable_auto_metadata
            )

        return type("BaseMapTransformCustom__" + name, (), {"__init__": ray_init, "__call__": ray_callable})

    @staticmethod
    def _process_ray(
        ray_input: dict[str, np.ndarray],
        name: str,
        f: Callable[[list[Document]], list[Document]],
        enable_auto_metadata: bool,
    ) -> dict[str, list]:
        # Have to do fully inline documents and metadata which means that we're forced to deserialize
        # metadata documents even though we just pass them through. If we instead had multiple columns,
        # we would have to make fake empty documents so that the doc and meta columns have the same number
        # of rows. Otherwise ray will raise an error.
        all_docs = [Document.deserialize(s) for s in ray_input.get("doc", [])]
        docs = [d for d in all_docs if not isinstance(d, MetadataDocument)]
        metadata = [d for d in all_docs if isinstance(d, MetadataDocument)]
        outputs = f(docs)
        if outputs is None:
            logging.warn(f"Function {name} returned nothing. If it has no outputs it should return an empty list")
            outputs = []
        elif isinstance(outputs, Document):
            outputs = [outputs]
        if not isinstance(outputs, list):
            raise ValueError(
                f"Function {name} returned {outputs} not the expected"
                " list of Document or the accepted single Document."
            )

        to_docs = [d for d in outputs if not isinstance(d, MetadataDocument)]
        if enable_auto_metadata and (len(docs) > 0 or len(to_docs) > 0):
            outputs.extend(update_lineage(docs, to_docs))
        outputs.extend(metadata)
        return {"doc": [d.serialize() for d in outputs]}


class CompositeTransform(UnaryNode):
    def __init__(self, child: Node, base_args: list[dict], enable_auto_metadata=True, **resource_args):
        super().__init__(child, **resource_args)
        self.nodes = CompositeTransform.combine(child, base_args, **resource_args)
        self._enable_auto_metadata = enable_auto_metadata

    @staticmethod
    def combine(last: Node, base_args: list[dict], **resource_args) -> list[BaseMapTransform]:
        nodes = []
        for a in base_args:
            args = resource_args | a
            last = BaseMapTransform(last, **args)
            nodes.append(last)

        return nodes

    def _local_process(self, in_docs: list[Document]) -> list[Document]:
        docs = in_docs
        for n in self.nodes:
            docs = n._local_process(docs)

        return docs

    def local_execute(self, all_docs: list[Document]) -> list[Document]:
        docs = [d for d in all_docs if not isinstance(d, MetadataDocument)]
        metadata = [d for d in all_docs if isinstance(d, MetadataDocument)]
        outputs = self._local_process(docs)
        to_docs = [d for d in outputs if not isinstance(d, MetadataDocument)]
        if self._enable_auto_metadata and (len(docs) > 0 or len(to_docs) > 0):
            outputs.extend(update_lineage(docs, to_docs))
        outputs.extend(metadata)
        return outputs

    def execute(self, **kwargs) -> "Dataset":
        from sycamore.executor import visit_parallelism

        for n in self.nodes:
            visit_parallelism(n)

        return self.nodes[-1].execute()
