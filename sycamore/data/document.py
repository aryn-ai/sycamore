from collections import UserDict
from typing import Any, Optional

from sycamore.data import BoundingBox, Element


class Document(UserDict):
    """
    A Document is a generic representation of an unstructured document in a format like PDF, HTML. Though different
    types of document may have different properties, they all contain the following common fields in Sycamore:
    """

    def __init__(self, document=None, /, **kwargs):
        if isinstance(document, bytes):
            from pickle import loads

            document = loads(document)
        super().__init__(document, **kwargs)

    @property
    def doc_id(self) -> Optional[str]:
        """A unique identifier for the document. Defaults to a uuid."""
        return self.data.get("doc_id")

    @doc_id.setter
    def doc_id(self, value: str) -> None:
        self.data["doc_id"] = value

    @property
    def type(self) -> Optional[str]:
        return self.data.get("type")

    @type.setter
    def type(self, value: str) -> None:
        self.data["type"] = value

    @property
    def text_representation(self) -> Optional[str]:
        return self.data.get("text_representation")

    @text_representation.setter
    def text_representation(self, value: str) -> None:
        self.data["text_representation"] = value

    @property
    def binary_representation(self) -> Optional[bytes]:
        """The raw content of the document in stored in the appropriate format.For example, the
        content of a PDF document will be stored as the binary_representation."""
        return self.data.get("binary_representation")

    @binary_representation.setter
    def binary_representation(self, value: bytes) -> None:
        self.data["binary_representation"] = value

    @binary_representation.deleter
    def binary_representation(self) -> None:
        self.data["binary_representation"] = None

    @property
    def elements(self) -> list[Element]:
        """A list of elements belonging to this document. A document does not necessarily always have
        elements, for instance, before a document is chunked."""
        return [Element(element) for element in self.data.get("elements", [])]

    @elements.setter
    def elements(self, elements: list[Element]):
        self.data["elements"] = [element.data for element in elements]

    @elements.deleter
    def elements(self) -> None:
        self.data["elements"] = []

    @property
    def embedding(self) -> Optional[list[float]]:
        return self.data.get("embedding")

    @embedding.setter
    def embedding(self, embedding: list[float]) -> None:
        self.data["embedding"] = embedding

    @property
    def parent_id(self) -> Optional[str]:
        """In Sycamore, certain operations create parent-child relationships between documents. For
        example, the explode transform promotes elements to be top-level documents, and these documents retain a
        pointer to the document from which they were created using the parent_id field. For those documents which
        have no parent, parent_id is None."""
        return self.data.get("parent_id")

    @parent_id.setter
    def parent_id(self, value: str) -> None:
        self.data["parent_id"] = value

    @property
    def bbox(self) -> Optional[BoundingBox]:
        return None if self.data.get("bbox") is None else BoundingBox(*self.data["bbox"])

    @bbox.setter
    def bbox(self, bbox: BoundingBox) -> None:
        self.data["bbox"] = bbox.coordinates

    @property
    def properties(self) -> dict[str, Any]:
        """A collection of system or customer defined properties, for instance, a PDF document might have
        title and author properties."""
        return self.data.get("properties", {})

    @properties.setter
    def properties(self, properties: dict[str, Any]):
        self.data["properties"] = properties

    @properties.deleter
    def properties(self) -> None:
        self.data["properties"] = {}

    def serialize(self) -> bytes:
        from pickle import dumps

        return dumps(self.data)

    @staticmethod
    def deserialize(raw: bytes) -> "Document":
        from pickle import loads

        return Document(loads(raw))

    @staticmethod
    def from_row(row: dict[str, bytes]) -> "Document":
        return Document(row["doc"])

    def to_row(self) -> dict[str, bytes]:
        return {"doc": self.serialize()}
