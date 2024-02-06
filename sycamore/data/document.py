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
        """Set the unique identifier of the document."""
        self.data["doc_id"] = value

    @property
    def type(self) -> Optional[str]:
        """The type of the document, e.g. pdf, html."""
        return self.data.get("type")

    @type.setter
    def type(self, value: str) -> None:
        """Set the type of the document."""
        self.data["type"] = value

    @property
    def text_representation(self) -> Optional[str]:
        """The text representation of the document."""
        return self.data.get("text_representation")

    @text_representation.setter
    def text_representation(self, value: str) -> None:
        """Set the text representation of the document."""
        self.data["text_representation"] = value

    @property
    def binary_representation(self) -> Optional[bytes]:
        """The raw content of the document stored in the appropriate format. For example, the
        content of a PDF document will be stored as the binary_representation."""
        return self.data.get("binary_representation")

    @binary_representation.setter
    def binary_representation(self, value: bytes) -> None:
        """Set the raw content of the document."""
        self.data["binary_representation"] = value

    @binary_representation.deleter
    def binary_representation(self) -> None:
        """Delete the raw content of the document."""
        self.data["binary_representation"] = None

    @property
    def elements(self) -> list[Element]:
        """A list of elements belonging to this document. A document does not necessarily always have
        elements, for instance, before a document is chunked."""
        return [Element(element) for element in self.data.get("elements", [])]

    @elements.setter
    def elements(self, elements: list[Element]):
        """Set the elements for this document."""
        self.data["elements"] = [element.data for element in elements]

    @elements.deleter
    def elements(self) -> None:
        """Delete the elements of this document."""
        self.data["elements"] = []

    @property
    def embedding(self) -> Optional[list[float]]:
        """Get the embedding for this document."""
        return self.data.get("embedding")

    @embedding.setter
    def embedding(self, embedding: list[float]) -> None:
        """Set the embedding for this document."""
        self.data["embedding"] = embedding

    @property
    def shingles(self) -> Optional[list[int]]:
        return self.data.get("shingles")

    @shingles.setter
    def shingles(self, shingles: list[int]) -> None:
        self.data["shingles"] = shingles

    @property
    def parent_id(self) -> Optional[str]:
        """In Sycamore, certain operations create parent-child relationships between documents. For
        example, the explode transform promotes elements to be top-level documents, and these documents retain a
        pointer to the document from which they were created using the parent_id field. For those documents which
        have no parent, parent_id is None."""
        return self.data.get("parent_id")

    @parent_id.setter
    def parent_id(self, value: str) -> None:
        """Set the parent_id for this document."""
        self.data["parent_id"] = value

    @property
    def bbox(self) -> Optional[BoundingBox]:
        """Get the bounding box for this document."""
        return None if self.data.get("bbox") is None else BoundingBox(*self.data["bbox"])

    @bbox.setter
    def bbox(self, bbox: BoundingBox) -> None:
        """Set the bounding box for this document."""
        self.data["bbox"] = bbox.coordinates

    @property
    def properties(self) -> dict[str, Any]:
        """A collection of system or customer defined properties, for instance, a PDF document might have
        title and author properties."""
        return self.data.get("properties", {})

    @properties.setter
    def properties(self, properties: dict[str, Any]):
        """Set all the proprites for this document."""
        self.data["properties"] = properties

    @properties.deleter
    def properties(self) -> None:
        """Delete all the properties of this document."""
        self.data["properties"] = {}

    def serialize(self) -> bytes:
        """Serialize this document to bytes."""
        from pickle import dumps

        return dumps(self.data)

    @staticmethod
    def deserialize(raw: bytes) -> "Document":
        """Unserialize from bytes to a Document."""
        from pickle import loads

        return Document(loads(raw))

    @staticmethod
    def from_row(row: dict[str, bytes]) -> "Document":
        """Unserialize a Ray row back into a Document."""
        return Document(row["doc"])

    def to_row(self) -> dict[str, bytes]:
        """Serialize this document into a row for use with Ray."""
        return {"doc": self.serialize()}


class OpenSearchQuery(Document):
    def __init__(
        self,
        document=None,
        **kwargs,
    ):
        super().__init__(document, **kwargs)
        self.data["type"] = "OpenSearchQuery"

    @property
    def query(self) -> Optional[dict[str, Any]]:
        """OpenSearch query body."""
        return self.data.get("query")

    @query.setter
    def query(self, value: dict[str, Any]) -> None:
        """Set the OpenSearch query body."""
        self.data["query"] = value

    @property
    def url_params(self) -> Optional[str]:
        """List of additional url parameters to send to the OpenSearch endpoint, e.g. index"""
        return self.data.get("url_params")

    @url_params.setter
    def url_params(self, value: str) -> None:
        """Set the list of additional url parameters to send to the OpenSearch endpoint."""
        self.data["url_params"] = value


class OpenSearchQueryResult(Document):
    def __init__(
        self,
        document=None,
        **kwargs,
    ):
        super().__init__(document, **kwargs)
        self.data["type"] = "OpenSearchQueryResult"

    @property
    def query(self) -> Optional[dict[str, Any]]:
        """The unmodified query used."""
        return self.data.get("query")

    @query.setter
    def query(self, value: dict[str, Any]) -> None:
        """Set the unmodified query."""
        self.data["query"] = value

    @property
    def hits(self) -> list[Element]:
        """List of documents retrieved by the query."""
        return self.data.get("hits", [])

    @hits.setter
    def hits(self, value: list[Element]) -> None:
        """Set the list of document retrieved."""
        self.data["hits"] = value

    @property
    def generated_answer(self) -> Optional[str]:
        """RAG generated answer."""
        return self.data.get("generated_answer")

    @generated_answer.setter
    def generated_answer(self, value: str) -> None:
        """Set the RAG generated answer."""
        self.data["generated_answer"] = value

    @property
    def result(self) -> Optional[str]:
        """Raw result from OpenSearch"""
        return self.data.get("result")

    @result.setter
    def result(self, value: str) -> None:
        """Set the raw result from OpenSearch."""
        self.data["result"] = value
