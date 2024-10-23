from sycamore.data import Document, Element
from sycamore.data.document import DocumentPropertyTypes
from sycamore.plan_nodes import Node, SingleThreadUser, NonGPUUser
from sycamore.transforms.map import Map
from sycamore.utils.markdown import elements_to_markdown


class Markdown(SingleThreadUser, NonGPUUser, Map):
    """
    The Markdown transform collapses all the elements of the Document
    into a single Element which is the Markdown representation of the
    entire Document, as the text_representation.

    Args:
        child: The source node or component that provides the documents
        kwargs: Optional resource-related arguments for the operation

    Example:
        .. code-block:: python

            md = Markdown(child=node)
            dataset = md.execute()
    """

    def __init__(self, child: Node, **kwargs):
        super().__init__(child, f=make_markdown, **kwargs)


def make_markdown(doc: Document) -> Document:
    elems = doc.elements
    text = elements_to_markdown(elems)
    pageset: set[int] = set()
    for elem in elems:
        pn = elem.properties.get(DocumentPropertyTypes.PAGE_NUMBER)
        if pn is not None:
            pageset.add(pn)
    pages = sorted(pageset)
    if not pages:
        pages.append(1)
    doc.elements = [
        Element(
            {
                "type": "Text",
                "bbox": (0.0, 0.0, 1.0, 1.0),  # best guess
                "properties": {
                    DocumentPropertyTypes.PAGE_NUMBER: pages[0],
                    "page_numbers": pages,
                },
                "text_representation": text,
                "binary_representation": text.encode(),
            }
        )
    ]
    return doc
