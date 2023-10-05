from typing import Callable

from ray.data import Dataset

from sycamore.plan_nodes import Node, NonGPUUser, NonCPUUser, Transform

from sycamore.data import Document
from sycamore.plan_nodes import UnaryNode
from sycamore.transforms.map import generate_map_batch_filter_function


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

    def execute(self) -> "Dataset":
        dataset = self.child().execute()
        return dataset.limit(self._limit)


class Filter(UnaryNode):
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
        super().__init__(child, **resource_args)
        self._f = f

    def execute(self) -> Dataset:
        input_dataset = self.child().execute()
        ray_callable = generate_map_batch_filter_function(self._f)
        return input_dataset.map_batches(ray_callable, **self.resource_args)
