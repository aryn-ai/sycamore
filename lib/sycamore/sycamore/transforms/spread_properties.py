from ray.data import Dataset

from sycamore.data import Document
from sycamore.plan_nodes import Node, Transform, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import generate_map_function
from sycamore.utils.time_trace import TimeTrace


class SpreadProperties(SingleThreadUser, NonGPUUser, Transform):
    """
    The SpreadProperties transform copies properties from each document to its
    subordinate elements.

    Args:
        child: The source node or component that provides the hierarchical documents to be exploded.
        resource_args: Additional resource-related arguments that can be passed to the explosion operation.

    Example:
        .. code-block:: python

            source_node = ...  # Define a source node or component that provides hierarchical documents.
            spread_transform = SpreadProperties(child=source_node, list=["title"])
            spread_dataset = spread_transform.execute()
    """

    def __init__(self, child: Node, props: list[str], **resource_args):
        super().__init__(child, **resource_args)
        self._props = props

    class SpreadPropertiesCallable:
        def __init__(self, props: list[str]):
            self._props = props

        def spreadProperties(self, parent: Document) -> Document:
            tt = TimeTrace("spreadProps")
            tt.start()
            newProps = {}
            for key in self._props:
                val = parent.properties.get(key)
                if val is not None:
                    newProps[key] = val

            # TODO: Have a way to let existing element properties win.
            for element in parent.elements:
                element.properties.update(newProps)
            tt.end()
            return parent

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        spreader = SpreadProperties.SpreadPropertiesCallable(self._props)
        return dataset.map(generate_map_function(spreader.spreadProperties))
