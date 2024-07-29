from collections import UserDict
import json
from typing import Any, Optional
import uuid

from sycamore.data import BoundingBox, Element
from sycamore.data.element import create_element


class Document(UserDict):
    """
    A Document is a generic representation of an unstructured document in a format like PDF, HTML. Though different
    types of document may have different properties, they all contain the following common fields in Sycamore:
    """

    def __init__(self, document=None, /, **kwargs):
        if isinstance(document, bytes):
            from pickle import loads

            document = loads(document)
            if "metadata" in document:
                raise ValueError("metadata must be deserialized with Document.deserialize not Document.__init__")

        super().__init__(document, **kwargs)
        if "properties" not in self.data:
            self.data["properties"] = {}

        if "elements" not in self.data or self.data["elements"] is None:
            self.data["elements"] = []
        elif not isinstance(self.data["elements"], list):
            raise ValueError("elements property should be a list")
        else:
            elements = self.data["elements"]
            for e in elements:
                if not (isinstance(e, dict) or isinstance(e, UserDict)):
                    raise ValueError(f"entries in elements property list must be dictionaries, not {type(e)}")
            self.data["elements"] = [create_element(**element) for element in self.data["elements"]]

        if "lineage_id" not in self.data:
            self.update_lineage_id()

    @property
    def doc_id(self) -> Optional[str]:
        """A unique identifier for the document. Defaults to None."""
        return self.data.get("doc_id")

    @doc_id.setter
    def doc_id(self, value: str) -> None:
        """Set the unique identifier of the document."""
        self.data["doc_id"] = value

    @property
    def lineage_id(self) -> str:
        """A unique identifier for the document in its lineage."""
        return self.data["lineage_id"]

    def update_lineage_id(self):
        """Update the lineage ID with a new identifier"""
        self.data["lineage_id"] = str(uuid.uuid4())

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
        return self.data["elements"]

    @elements.setter
    def elements(self, elements: list[Element]):
        """Set the elements for this document."""
        self.data["elements"] = elements

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
        return self.data["properties"]

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

        data = loads(raw)
        if "metadata" in data:
            return MetadataDocument(data)
        else:
            return Document(data)

    @staticmethod
    def from_row(row: dict[str, bytes]) -> "Document":
        """Unserialize a Ray row back into a Document."""
        return Document.deserialize(row["doc"])

    def to_row(self) -> dict[str, bytes]:
        """Serialize this document into a row for use with Ray."""
        return {"doc": self.serialize()}

    def __str__(self) -> str:
        """Return a pretty-printed string representing this document."""
        d = {
            "doc_id": self.doc_id,
            "lineage_id": self.lineage_id,
            "type": self.type,
            "text_representation": self.text_representation[0:40] + "..." if self.text_representation else None,
            "binary_representation": (
                f"<{len(self.binary_representation)} bytes>" if self.binary_representation else None
            ),
            "elements": [str(e) for e in self.elements],
            "embedding": (str(self.embedding[0:4]) + f"... <{len(self.embedding)} total>") if self.embedding else None,
            "shingles": (str(self.shingles[0:4]) + f"... <{len(self.shingles)} total>") if self.shingles else None,
            "parent_id": self.parent_id,
            "bbox": str(self.bbox),
            "properties": self.properties,
        }
        return json.dumps(d, indent=2)
    
    def field_to_value(self, field: str) -> Any:
        """
        Extracts the value for a particular document field.

        Args:
            doc: The document
            field: The field in dotted notation to indicate nesting, e.g. doc.properties.schema.

        Returns:
            The value associated with the document field.
        """
        fields = field.split(".")
        value = getattr(self, fields[0])
        if len(fields) > 1:
            assert fields[0] == "properties"
            for f in fields[1:]:
                value = value[f]
        return value


class MetadataDocument(Document):
    def __init__(self, document=None, **kwargs):
        super().__init__(document)
        if "metadata" not in self.data:
            self.data["metadata"] = {}
        self.data["metadata"].update(kwargs)
        del self.data["lineage_id"]
        del self.data["elements"]

    # Override some of the common operations to make it hard to mis-use metadata. If any of these
    # are called it means that something tried to process a MetadataDocument as if it was a
    # Document.

    @property
    def doc_id(self) -> Optional[str]:
        """A unique identifier for the document. Defaults to a uuid."""
        raise ValueError("MetadataDocument does not have doc_id")

    @doc_id.setter
    def doc_id(self, value: str) -> None:
        """Set the unique identifier of the document."""
        raise ValueError("MetadataDocument does not have doc_id")

    @property
    def lineage_id(self) -> str:
        """A unique identifier for the document in its lineage."""
        raise ValueError("MetadataDocument does not have lineage_id")

    @lineage_id.setter
    def lineage_id(self, value: str) -> None:
        """Set the unique identifier for the document in its lineage."""
        raise ValueError("MetadataDocument does not have lineage_id")

    @property
    def text_representation(self):
        raise ValueError("MetadataDocument does not have text_representation")

    @text_representation.setter
    def text_representation(self, value: str) -> None:
        raise ValueError("MetadataDocument does not have text_representation")

    @property
    def binary_representation(self):
        raise ValueError("MetadataDocument does not have binary_representation")

    @binary_representation.setter
    def binary_representation(self, value: bytes) -> None:
        raise ValueError("MetadataDocument does not have binary_representation")

    @property
    def elements(self) -> list[Element]:
        raise ValueError("MetadataDocument does not have elements")

    @elements.setter
    def elements(self, elements: list[Element]):
        raise ValueError("MetadataDocument does not have elements")

    @property
    def properties(self):
        raise ValueError("MetadataDocument does not have properties")

    @properties.setter
    def properties(self, properties: dict[str, Any]):
        raise ValueError("MetadataDocument does not have properties")

    @property
    def metadata(self) -> dict[str, Any]:
        """Internal metadata about processing."""
        return self.data.get("metadata", {})

    @metadata.setter
    def metadata(self, metadata: dict[str, Any]):
        """Set all the properties for this document."""
        self.data["metadata"] = metadata

    @metadata.deleter
    def metadata(self) -> None:
        """Delete all the metadata of this document."""
        self.data["metadata"] = {}


def split_data_metadata(all: list[Document]) -> tuple[list[Document], list[MetadataDocument]]:
    return (
        [d for d in all if not isinstance(d, MetadataDocument)],
        [d for d in all if isinstance(d, MetadataDocument)],
    )


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
    def index(self) -> Optional[str]:
        """OpenSearch index."""
        return self.data.get("index")

    @index.setter
    def index(self, value: str) -> None:
        """Set the OpenSearch index."""
        self.data["index"] = value

    @property
    def params(self) -> Optional[dict[str, Any]]:
        """Dict of additional parameters to send to the OpenSearch endpoint."""
        return self.data.get("params")

    @params.setter
    def params(self, value: dict[str, Any]) -> None:
        """Set the list of additional parameters to send to the OpenSearch endpoint."""
        self.data["params"] = value

    @property
    def headers(self) -> Optional[dict[str, Any]]:
        """Dict of additional headers to send to the OpenSearch endpoint."""
        return self.data.get("headers")

    @headers.setter
    def headers(self, value: dict[str, Any]) -> None:
        """Set the list of additional headers to send to the OpenSearch endpoint."""
        self.data["headers"] = value

    @staticmethod
    def deserialize(raw: bytes) -> "OpenSearchQuery":
        """Deserialize from bytes to a OpenSearchQuery."""
        from pickle import loads

        return OpenSearchQuery(loads(raw))


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
    def result(self) -> Optional[Any]:
        """Raw result from OpenSearch"""
        return self.data.get("result")

    @result.setter
    def result(self, value: Any) -> None:
        """Set the raw result from OpenSearch."""
        self.data["result"] = value

    @staticmethod
    def deserialize(raw: bytes) -> "OpenSearchQueryResult":
        """Deserialize from bytes to a OpenSearchQueryResult."""
        from pickle import loads

        return OpenSearchQueryResult(loads(raw))
