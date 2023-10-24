from typing import Any, Callable, Iterable, Optional

from ray.data import ActorPoolStrategy, Dataset

from sycamore.data import Document
from sycamore.plan_nodes import Node, UnaryNode

from sycamore.utils import (
    generate_map_function,
    generate_map_class,
    generate_flat_map_function,
    generate_flat_map_class,
    generate_map_batch_function,
    generate_map_batch_class,
)


class Map(UnaryNode):
    """
    Map is a transformation class for applying a callable function to each document in a dataset.

    Example:
         .. code-block:: python

            def custom_mapping_function(document: Document) -> Document:
                # Custom logic to transform the document
                return transformed_document

            map_transformer = Map(input_dataset_node, f=custom_mapping_function)
            transformed_dataset = map_transformer.execute()
    """

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
    """
    FlatMap is a transformation class for applying a callable function to each document in a dataset and flattening
    the resulting list of documents.

    Example:
         .. code-block:: python

            def custom_flat_mapping_function(document: Document) -> list[Document]:
                # Custom logic to transform the document and return a list of documents
                return [transformed_document_1, transformed_document_2]

            flat_map_transformer = FlatMap(input_dataset_node, f=custom_flat_mapping_function)
            flattened_dataset = flat_map_transformer.execute()

    """

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
    """
    The MapBatch transform is similar to Map, except that it processes a list of documents and returns a list of
    documents. MapBatches is ideal for transformations that get performance benefits from batching.

    Example:
         .. code-block:: python

            def custom_map_batch_function(documents: list[Document]) -> list[Document]:
                # Custom logic to transform the documents
                return transformed_documents

            map_transformer = Map(input_dataset_node, f=custom_map_batch_function)
            transformed_dataset = map_transformer.execute()
    """

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
