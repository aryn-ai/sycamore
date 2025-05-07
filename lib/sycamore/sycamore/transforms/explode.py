from typing import Union, Optional, Callable

import streamlit

from sycamore.data import Document, HierarchicalDocument, mkdocid
from sycamore.data.element import TableElement, Element
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import FlatMap
from sycamore.utils.time_trace import timetrace


class Explode(SingleThreadUser, NonGPUUser, FlatMap):
    """
    The Explode transform converts the elements of each document into top-level documents. For example, if you explode a
    DocSet with a single document containing two elements, the resulting DocSet will have three documents - the original
    plus a new Document for each of the elements.

    Args:
        child: The source node or component that provides the hierarchical documents to be exploded.
        resource_args: Additional resource-related arguments that can be passed to the explosion operation.

    Example:
        .. code-block:: python

            source_node = ...  # Define a source node or component that provides hierarchical documents.
            explode_transform = Explode(child=source_node)
            exploded_dataset = explode_transform.execute()
    """

    def __init__(self, child: Node, **resource_args):
        super().__init__(child, f=Explode.explode, **resource_args)

    @staticmethod
    @timetrace("explode")
    def explode(parent: Union[Document, HierarchicalDocument], **kwargs) -> Union[list[Document], list[HierarchicalDocument]]:
        if isinstance(parent, HierarchicalDocument):
            return Explode.explode_hierarchical(parent)
        if isinstance(parent, Document):
            return Explode.explode_default(parent, **kwargs)
        parent.properties["exploded"] = True
        raise ValueError(f"Unsupported document type: {type(parent)}")

    @staticmethod
    @timetrace("explode")
    def explode_default(parent: Document, doc_id_func: Optional[Callable[[Element], str]] = None,
                        spread_properties_func: Optional[Callable[[str], bool]] = lambda p: p.startswith("_")) -> list[Document]:
        documents: list[Document] = [parent]
        for i, element in enumerate(parent.elements):
            cur = Document(element.data)
            cur.doc_id = doc_id_func(element) if doc_id_func else mkdocid("c")
            cur.parent_id = parent.doc_id
            if isinstance(element, TableElement):
                cur.text_representation = element.text_representation
            for doc_property in parent.properties.keys():
                if spread_properties_func(doc_property):
                    cur.properties[doc_property] = parent.properties[doc_property]
            documents.append(cur)
        del parent.elements
        return documents

    @staticmethod
    @timetrace("explode")
    def explode_hierarchical(parent: HierarchicalDocument) -> list[HierarchicalDocument]:
        documents: list[HierarchicalDocument] = [parent]
        for document in parent.children:
            documents.extend(Explode.explode_hierarchical(document))

        del parent.children
        return documents
