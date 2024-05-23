from sycamore.data import Document
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace


class SpreadProperties(SingleThreadUser, NonGPUUser, Map):
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
        super().__init__(child, f=SpreadProperties.spread_properties, args=[props], **resource_args)

    @staticmethod
    @timetrace("spreadProps")
    def spread_properties(parent: Document, props) -> Document:
        newProps = {}
        for key in props:
            val = parent.properties.get(key)
            if val is not None:
                newProps[key] = val

        # TODO: Have a way to let existing element properties win.
        for element in parent.elements:
            element.properties.update(newProps)
        return parent
