from typing import Any, Callable, Iterable, Optional


from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.transforms.base import BaseMapTransform


class Map(BaseMapTransform):
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
        super().__init__(child, f=Map.wrap(f), name=f.__name__, **resource_args)

    @staticmethod
    def wrap(f: Callable[[Document], Document]) -> Callable[[list[Document]], list[Document]]:
        if isinstance(f, type):
            # mypy doesn't understand the dynamic class inheritence.
            class _Wrap(f):  # type: ignore[valid-type,misc]
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def __call__(self, docs):
                    s = super()
                    return [s.__call__(d) for d in docs]

            return _Wrap
        else:

            def _wrap(docs):
                return [f(d) for d in docs]

            return _wrap


class FlatMap(BaseMapTransform):
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
        super().__init__(child, f=FlatMap.wrap(f), name=f.__name__, **resource_args)

    @staticmethod
    def wrap(f: Callable[[Document], list[Document]]) -> Callable[[list[Document]], list[Document]]:
        if isinstance(f, type):

            class _Wrap(f):  # type: ignore[valid-type,misc]
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)

                def __call__(self, docs):
                    assert isinstance(docs, list)
                    s = super()
                    ret = []
                    for d in docs:
                        assert isinstance(d, Document)
                        ret.extend(s.__call__(d))
                    return ret

            return _Wrap
        else:

            def _wrap(docs):
                assert isinstance(docs, list)
                ret = []
                for d in docs:
                    assert isinstance(d, Document)
                    o = f(d)
                    ret.extend(o)
                return ret

            return _wrap


class MapBatch(BaseMapTransform):
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
        super().__init__(
            child,
            f=f,
            args=f_args,
            kwargs=f_kwargs,
            constructor_args=f_constructor_args,
            constructor_kwargs=f_constructor_kwargs,
            **resource_args
        )
