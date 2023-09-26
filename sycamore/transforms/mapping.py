from collections import defaultdict
from typing import Any, Callable, Iterable, Optional, Type

import numpy as np
from ray.data import ActorPoolStrategy, Dataset

from sycamore.data import Document
from sycamore.plan_nodes import Node, UnaryNode


def generate_map_function(f: Callable[[Document], Document]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def ray_callable(input_dict: dict[str, Any]) -> dict[str, Any]:
        document = f(Document(input_dict))
        return document.data

    return ray_callable


def generate_map_class(c: Type[Callable[[Document], Document]]) -> Type[Callable[[dict[str, Any]], dict[str, Any]]]:
    def ray_init(self):
        self.base = c()

    def ray_callable(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        document = self.base(Document(input_dict))
        return document.data

    new_class = type("CustomRay" + c.__name__, (), {"__init__": ray_init, "__call__": ray_callable})
    return new_class


def generate_flat_map_function(
    f: Callable[[Document], list[Document]]
) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
    def ray_callable(input_dict: dict[str, Any]) -> list[dict[str, Any]]:
        documents = f(Document(input_dict))
        return [document.data for document in documents]

    return ray_callable


def generate_flat_map_class(
    c: Type[Callable[[Document], list[Document]]]
) -> Type[Callable[[dict[str, Any]], list[dict[str, Any]]]]:
    def ray_init(self):
        self.base = c()

    def ray_callable(self, input_dict: dict[str, Any]) -> list[dict[str, Any]]:
        documents = self.base(Document(input_dict))
        return [document.data for document in documents]

    new_class = type("CustomRay" + c.__name__, (), {"__init__": ray_init, "__call__": ray_callable})
    return new_class


def generate_map_batch_function(
    f: Callable[[list[Document]], list[Document]]
) -> Callable[[dict[str, np.ndarray]], dict[str, list]]:
    def ray_callable(doc_batch: dict[str, np.ndarray]) -> dict[str, list]:
        input_docs = _get_documents_from_columnar_format(doc_batch)
        output_docs = f(input_docs)

        return _get_columnar_format_from_documents(output_docs)

    return ray_callable


def generate_map_batch_filter_function(
    f: Callable[[Document], bool]
) -> Callable[[dict[str, np.ndarray]], dict[str, list]]:
    def ray_callable(doc_batch: dict[str, np.ndarray]) -> dict[str, list]:
        input_docs = _get_documents_from_columnar_format(doc_batch)
        output_docs = list(filter(f, input_docs))

        return _get_columnar_format_from_documents(output_docs)

    return ray_callable


def _get_documents_from_columnar_format(doc_batch: dict[str, np.ndarray]) -> list[Document]:
    input_docs = []
    cols = doc_batch.keys()
    rows = doc_batch.values()

    for row in zip(*rows):
        document = {}
        for i, col in enumerate(cols):
            document[col] = row[i]
        input_docs.append(Document(document))

    return input_docs


def _get_columnar_format_from_documents(doc_batch: list[Document]) -> dict[str, list]:
    output = defaultdict(list)

    for doc in doc_batch:
        for key, value in doc.data.items():
            output[key].append(value)

    return output


def generate_map_batch_class(
    c: Type[Callable[[list[Document]], list[Document]]],
    f_args: Optional[Iterable[Any]] = None,
    f_kwargs: Optional[dict[str, Any]] = None,
    f_constructor_args: Optional[Iterable[Any]] = None,
    f_constructor_kwargs: Optional[dict[str, Any]] = None,
) -> Type[Callable[[list[dict[str, Any]]], list[dict[str, Any]]]]:
    if f_constructor_args is None:
        f_constructor_args = tuple()
    if f_constructor_kwargs is None:
        f_constructor_kwargs = {}
    if f_args is None:
        f_args = tuple()
    if f_kwargs is None:
        f_kwargs = {}

    def ray_init(self):
        self.base = c(*f_constructor_args, **f_constructor_kwargs)

    def ray_callable(self, doc_batch: dict[str, np.ndarray]) -> dict[str, list]:
        input_docs = _get_documents_from_columnar_format(doc_batch)
        output_docs = self.base(input_docs, *f_args, **f_kwargs)

        return _get_columnar_format_from_documents(output_docs)

    new_class = type("CustomRay" + c.__name__, (), {"__init__": ray_init, "__call__": ray_callable})
    return new_class


def generate_map_batch_class_from_callable(
    f: Callable[[list[Document]], list[Document]],
) -> Type[Callable[[list[dict[str, Any]]], list[dict[str, Any]]]]:
    def ray_callable(self, doc_batch: dict[str, np.ndarray]) -> dict[str, list]:
        input_docs = _get_documents_from_columnar_format(doc_batch)
        output_docs = f(input_docs)

        return _get_columnar_format_from_documents(output_docs)

    new_class = type("CustomRay", (), {"__call__": ray_callable})
    return new_class


class Map(UnaryNode):
    def __init__(self, child: Node, *, f: Callable[[Document], Document], **resource_args):
        super().__init__(child, **resource_args)
        self._f = f

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        if isinstance(self._f, type):
            ray_callable = generate_map_class(self._f)
            return input_dataset.map(ray_callable, compute=ActorPoolStrategy(size=1), **self.resource_args)
        else:
            ray_callable = generate_map_function(self._f)
            return input_dataset.map(ray_callable, **self.resource_args)


class FlatMap(UnaryNode):
    def __init__(self, child: Node, *, f: Callable[[Document], list[Document]], **resource_args):
        super().__init__(child, **resource_args)
        self._f = f

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        if isinstance(self._f, type):
            ray_callable = generate_flat_map_class(self._f)
            return input_dataset.flat_map(ray_callable, compute=ActorPoolStrategy(size=1), **self.resource_args)
        else:
            ray_callable = generate_flat_map_function(self._f)
            return input_dataset.flat_map(ray_callable, **self.resource_args)


class MapBatch(UnaryNode):
    def __init__(
        self,
        child: Node,
        *,
        f: Callable[[list[Document]], list[Document]],
        f_args: Optional[Iterable[Any]] = None,
        f_kwargs: Optional[dict[str, Any]] = None,
        f_constructor_args: Optional[Iterable[Any]] = None,
        f_constructor_kwargs: Optional[dict[str, Any]] = None,
        **resource_args
    ):
        super().__init__(child, **resource_args)
        self._f = f
        self._f_args = f_args
        self._f_kwargs = f_kwargs
        self._f_constructor_args = f_constructor_args
        self._f_constructor_kwargs = f_constructor_kwargs

    def execute(self) -> "Dataset":
        input_dataset = self.child().execute()
        if isinstance(self._f, type):
            ray_callable = generate_map_batch_class(
                self._f, self._f_args, self._f_kwargs, self._f_constructor_args, self._f_constructor_kwargs
            )
            return input_dataset.map_batches(ray_callable, compute=ActorPoolStrategy(size=1), **self.resource_args)
        else:
            ray_callable = generate_map_batch_function(self._f)
            return input_dataset.map_batches(ray_callable, **self.resource_args)
