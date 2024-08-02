from sycamore.data import Document
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
import json
from typing import List, Dict


class AssignDocProperties(SingleThreadUser, NonGPUUser, Map):
    """
    The AssignDocProperties transform is used to copy properties from specific element stored
    within a JSON string to the parent document. This allows for the consolidation of key
    attributes at the document level.

    Args:
        child: The source node or component that provides the hierarchical documents to be exploded.
        resource_args: Additional resource-related arguments that can be passed to the explosion operation.

    Example:
        .. code-block:: python

            source_node = ...  # Define a source node or component that provides hierarchical documents.
            property_transform = AssignDocProperties(child=source_node, list=["table"])
            property_dataset = property_transform.execute()
    """

    def __init__(self, child: Node, parameters: List[str], **resource_args):
        super().__init__(child, f=AssignDocProperties.assign_doc_properties, args=parameters, **resource_args)

    @staticmethod
    def _parse_json(s: str) -> Dict:
        s = s[next(idx for idx, c in enumerate(s) if c in "{[") :]
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            return json.loads(s[: e.pos])

    @timetrace("AssignProps")
    @staticmethod
    def assign_doc_properties(parent: Document, element_type: str, property_name: str) -> Document:
        # element count is zero indexed
        assert property_name is not None
        for e in parent.elements:
            if e.type == element_type:
                property_str = e.properties.get(property_name)
                assert isinstance(property_str, str), f"Expected str, got {type(property_str).__name__}"
                property_value = AssignDocProperties._parse_json(property_str)
                e.properties.update(property_value)
                parent.properties["entity"] = property_value
                break

        return parent
