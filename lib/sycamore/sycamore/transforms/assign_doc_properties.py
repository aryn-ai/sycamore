from sycamore.data import Document
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.time_trace import timetrace
import logging
import json

class AssignDocProperties(SingleThreadUser, NonGPUUser, Map):
    """
    The assign_docset_properties transform copies properties from specific element to 
    parent document.

    Args:
        child: The source node or component that provides the hierarchical documents to be exploded.
        resource_args: Additional resource-related arguments that can be passed to the explosion operation.

    Example:
        .. code-block:: python

            source_node = ...  # Define a source node or component that provides hierarchical documents.
            spread_transform = SpreadProperties(child=source_node, list=["table"])
            spread_dataset = spread_transform.execute()
    """

    def __init__(self, child: Node, element_type: str, **resource_args):
        super().__init__(child, f=AssignDocProperties.assign_doc_properties, args=[element_type], **resource_args)


    def _parse_json_garbage(s):
        s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            return json.loads(s[:e.pos])
    @staticmethod
    @timetrace("AssignProps")
    def assign_doc_properties(parent: Document, element_type) -> Document:
        element_count = 0
        elementPro = "llm_response"
        for e in parent.elements:
            if e.type == element_type:
                e.properties.update(AssignDocProperties._parse_json_garbage(e.properties.get(elementPro)))
                parent.properties["entity"] = e.properties.copy()
                break

        return parent
