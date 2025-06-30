from typing import Callable, Optional

from sycamore.data import Document, Element
from sycamore.data.document import DocumentPropertyTypes, DocumentSource


class DocumentReconstructor:
    def __init__(
        self,
        index_name: str,
        reconstruct_fn: Optional[Callable[[str, str], Document]] = None,
    ):
        self.index_name = index_name
        self.reconstruct_fn = reconstruct_fn

    def get_required_source_fields(self) -> list[str]:
        return ["parent_id"]

    def get_doc_id(self, data: dict) -> str:
        return data["_source"]["parent_id"] or data["_id"]

    def reconstruct(self, output: list[dict]) -> list[Document]:
        result: list[Document] = []
        if not self.reconstruct_fn:
            raise ValueError("Reconstruct function is not defined.")
        unique = set()
        for data in output:
            doc_id = self.get_doc_id(data)
            if doc_id not in unique:
                result.append(self.reconstruct_fn(self.index_name, self.get_doc_id(data)))
                unique.add(doc_id)
        return result


class RAGDocumentReconstructor(DocumentReconstructor):
    def __init__(
        self,
        index_name: str,
    ):
        super().__init__(index_name, reconstruct_fn=None)

    def get_required_source_fields(self) -> list[str]:
        fields = [
            "doc_id",
            "parent_id",
            "properties",
            "type",
            "text_representation",
        ]
        return fields

    def reconstruct(self, output: list[dict]) -> list[Document]:
        result: list[Document] = []
        unique_docs: dict[str, Document] = {}

        for element in output:
            doc = Document(
                {
                    **element.get("_source", {}),
                }
            )
            doc.properties[DocumentPropertyTypes.SOURCE] = DocumentSource.DB_QUERY
            doc.properties["search_relevance_score"] = element["_score"]
            assert doc.doc_id, "Retrieved invalid doc with a missing doc_id"
            if not doc.parent_id:
                temp = unique_docs[doc.doc_id].elements if doc.doc_id in unique_docs else []
                unique_docs[doc.doc_id] = doc
                parent = unique_docs[doc.doc_id]
                parent.elements = temp
            else:
                unique_docs[doc.parent_id] = unique_docs.get(
                    doc.parent_id,
                    Document(
                        {
                            "doc_id": doc.parent_id,
                            "properties": {
                                **doc.properties,
                                DocumentPropertyTypes.SOURCE: DocumentSource.DOCUMENT_RECONSTRUCTION_PARENT,
                            },
                            "type": doc.type,
                        }
                    ),
                )
                parent = unique_docs[doc.parent_id]
                parent.elements.append(Element(doc.data))

        result = list(unique_docs.values())
        return result
