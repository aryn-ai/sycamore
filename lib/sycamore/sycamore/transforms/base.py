import logging
from typing import Any, Callable, Iterable, Optional

import numpy as np
from ray.data import ActorPoolStrategy, Dataset

from sycamore.data import Document, MetadataDocument
from sycamore.data.document import split_data_metadata
from sycamore.plan_nodes import Node, UnaryNode


def take_separate(dataset: Dataset, limit: Optional[int] = None) -> tuple[list[Document], list[MetadataDocument]]:
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


class BaseMapTransform(UnaryNode):
    """
    BaseMapTransform abstracts away MetadataDocuments from all other transforms.

    If f is a class type, the class will be instantiated and run as an actor in ray.
    If f is an object type and resource_args["compute"] is set to ActorPoolStrategy, it will run as an actor
    Otherwise f will be run as a function.
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
        if child is None:
            logging.info("Assuming this is for local execution only, not checking serializability")
        else:
            # If serializability fails, the error messages are very confusing. These checks
            # give a much more sensible error message and give it before ray starts execution.
            from ray.util import inspect_serializability
            import io

            log = io.StringIO()
            (ok, s) = inspect_serializability(
                [f, name, args, kwargs, constructor_args, constructor_kwargs], print_file=log
            )
            if not ok:
                raise ValueError(f"Something for {name} isn't serializable {s}\nLog: {log.getvalue()}")

        if isinstance(f, type) and "compute" not in resource_args:
            # classes require actor strategy for now
            resource_args["compute"] = ActorPoolStrategy(size=1)

        super().__init__(child, **resource_args)
        if name is None:
            if "__name__" in dir(f):
                name = f.__name__
            elif "__class__" in dir(f):
                name = f.__class__.__name__
            else:
                raise ValueError(f"Unable to extract name from {f}, all members: {dir(f)}")

        self._f = f
        self._name = name
        self._args = args
        self._kwargs = kwargs
        self._constructor_args = constructor_args
        self._constructor_kwargs = constructor_kwargs
        self._enable_auto_metadata = enable_auto_metadata

    def execute(self) -> "Dataset":
        if "num_gpus" in self.resource_args:
            assert self.resource_args["num_gpus"] > 0

        input_dataset = self.child().execute()

        if isinstance(self._f, type):  # is f a class?
            # Maybe add a class as function variant if the caller specified TaskPoolStrategy
            return input_dataset.map_batches(self._map_class(), **self.resource_args)
        elif "compute" in self.resource_args and isinstance(self.resource_args["compute"], ActorPoolStrategy):
            # Ray requires a class for ActorPoolStrategy.
            return input_dataset.map_batches(self._map_callable_as_class(), **self.resource_args)
        else:
            return input_dataset.map_batches(self._map_function(), **self.resource_args)

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
        if enable_auto_metadata:
            outputs.extend(BaseMapTransform._update_lineage(docs, to_docs))
        outputs.extend(metadata)
        return {"doc": [d.serialize() for d in outputs]}

    @classmethod
    def _update_lineage(cls, from_docs, to_docs):
        from_ids = [d.lineage_id for d in from_docs]
        for d in to_docs:
            d.update_lineage_id()
        to_ids = [d.lineage_id for d in to_docs]

        return [MetadataDocument(lineage_links={"from_ids": from_ids, "to_ids": to_ids})]


class CompositeTransform(UnaryNode):
    def __init__(self, child: Node, base_args: list[dict], **resource_args):
        super().__init__(child, **resource_args)
        self.nodes = CompositeTransform.combine(child, base_args, **resource_args)

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

    def execute(self) -> Dataset:
        return self.nodes[-1].execute()
