import logging
from typing import Any, Callable, Iterable, Optional

import numpy as np
from ray.data import ActorPoolStrategy, Dataset

from sycamore.data import Document, MetadataDocument
from sycamore.plan_nodes import Node, UnaryNode


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
    """

    def __init__(
        self,
        child: Node,
        *,
        f: Callable[[list[Document]], list[Document]],
        name: Optional[str] = None,
        args: Optional[Iterable[Any]] = None,
        kwargs: Optional[dict[str, Any]] = None,
        constructor_args: Optional[Iterable[Any]] = None,
        constructor_kwargs: Optional[dict[str, Any]] = None,
        **resource_args,
    ):
        super().__init__(child, **resource_args)
        if name is None:
            if isinstance(f, type):
                name = type(f).__name__
            else:
                name = f.__name__

        self._f = f
        self._name = name
        self._args = args
        self._kwargs = kwargs
        self._constructor_args = constructor_args
        self._constructor_kwargs = constructor_kwargs

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()

        if isinstance(self._f, type):
            return input_dataset.map_batches(self._map_class(), compute=ActorPoolStrategy(size=1), **self.resource_args)
        else:
            return input_dataset.map_batches(self._map_function(), **self.resource_args)

    def _map_function(self):
        f = self._f
        name = self._name
        args = _noneOr(self._args, tuple())
        kwargs = _noneOr(self._kwargs, {})

        @rename(name)
        def ray_callable(ray_input: dict[str, np.ndarray]) -> dict[str, list]:
            return BaseMapTransform._process_ray(ray_input, name, lambda d: f(d, *args, **kwargs))

        return ray_callable

    def _map_class(self):
        c = self._f
        name = self._name
        args = _noneOr(self._args, tuple())
        kwargs = _noneOr(self._kwargs, {})
        c_args = _noneOr(self._constructor_args, tuple())
        c_kwargs = _noneOr(self._constructor_kwargs, {})

        def ray_init(self):
            self.base = c(*c_args, **c_kwargs)

        def ray_callable(self, ray_input: dict[str, np.ndarray]) -> dict[str, list]:
            return BaseMapTransform._process_ray(ray_input, name, lambda d: self.base(d, *args, **kwargs))

        return type("BaseMapTransformCustom__" + name, (), {"__init__": ray_init, "__call__": ray_callable})

    @staticmethod
    def _process_ray(
        ray_input: dict[str, np.ndarray], name: str, f: Callable[[list[Document]], list[Document]]
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
