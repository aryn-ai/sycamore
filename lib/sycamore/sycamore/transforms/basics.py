from typing import Callable, TYPE_CHECKING

from sycamore.data import Document
from sycamore.plan_nodes import Node, NonGPUUser, NonCPUUser, Transform
from sycamore.transforms.map import MapBatch

if TYPE_CHECKING:
    from ray.data import Dataset


class Limit(NonCPUUser, NonGPUUser, Transform):
    """
    Limit is a transformation that restricts the size of a dataset to a specified number of records.

    Args:
        child: The source node or component that provides the dataset to be limited.
        limit: The maximum number of records to include in the resulting dataset.

    Example:
         .. code-block:: python

            source_node = ...  # Define a source node or component that provides a dataset.
            limit_transform = Limit(child=source_node, limit=100)
            limited_dataset = limit_transform.execute()
    """

    def __init__(self, child: Node, limit: int):
        super().__init__(child)
        self._limit = limit

    def execute(self, **kwargs) -> "Dataset":
        dataset = self.child().execute()
        return dataset.limit(self._limit)

    def local_execute(self, all_docs: list[Document]):
        return all_docs[0 : self._limit]


class Filter(MapBatch):
    """
    Filter is a transformation that applies a user-defined filter function to a dataset.

    Args:
        child: The source node or component that provides the dataset to be filtered.
        f: A callable function that takes a Document object and returns a boolean indicating whether the document
            should be included in the filtered dataset.
        resource_args: Additional resource-related arguments that can be passed to the filtering operation.

    Example:
         .. code-block:: python

            source_node = ...  # Define a source node or component that provides a dataset.
            def custom_filter(doc: Document) -> bool:
                # Define your custom filtering logic here.
                return doc.some_property == some_value

            filter_transform = Filter(child=source_node, f=custom_filter)
            filtered_dataset = filter_transform.execute()

    """

    def __init__(self, child: Node, *, f: Callable[[Document], bool], **resource_args):
        super().__init__(child, f=lambda docs: [d for d in docs if f(d)], **resource_args)
