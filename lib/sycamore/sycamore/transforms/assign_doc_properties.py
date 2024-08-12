from sycamore.data import Document
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
from typing import List, Dict


class AssignDocProperties(SingleThreadUser, NonGPUUser, Map):
    """
    The AssignDocProperties transform is used to copy properties from first element pf a specific type
    to the parent document. This allows for the consolidation of key attributes at the document level.

    Args:
        child: The source node or component that provides the dataset for assigning properties from element.
        resource_args: Additional resource-related arguments passed to the operation for property assignment.

    Example:
        .. code-block:: python

            source_node = ...  # Define a source node or component that provides hierarchical documents.
            property_transform = AssignDocProperties(child=source_node, list=["table", "llm_response"])
            property_dataset = property_transform.execute()
    """

    def __init__(self, child: Node, parameters: List[str], **resource_args):
        super().__init__(child, f=AssignDocProperties.assign_doc_properties, args=parameters, **resource_args)

    @staticmethod
    @timetrace("AssignProps")
    def assign_doc_properties(parent: Document, element_type: str, property_name: str) -> Document:
        # element count is zero indexed
        assert property_name is not None
        for e in parent.elements:
            if e.type == element_type and property_name in e.properties.keys():
                property = e.properties.get(property_name)
                assert isinstance(property, Dict), f"Expected Dict, got {type(property).__name__}"
                parent.properties["entity"] = property
                break

        return parent
